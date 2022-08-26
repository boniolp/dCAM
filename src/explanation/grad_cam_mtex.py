from torch import topk
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch
from typing import cast, Union, List





class _BaseWrapper(object):
	def __init__(self, model):
		super(_BaseWrapper, self).__init__()
		self.device = next(model.parameters()).device
		self.model = model
		self.handlers = []

	def _encode_one_hot(self, ids):
		one_hot = torch.zeros_like(self.logits).to(self.device)
		one_hot.scatter_(1, ids, 1.0)
		return one_hot

	def forward(self, image):
		self.image_shape = image.shape[2:]
		self.logits = self.model(image)
		self.probs = F.softmax(self.logits, dim=1)
		return self.probs.sort(dim=1, descending=True)

	def backward(self, ids):
		one_hot = self._encode_one_hot(ids)
		self.model.zero_grad()
		self.logits.backward(gradient=one_hot, retain_graph=True)

	def generate(self):
		raise NotImplementedError

	def remove_hook(self):
		for handle in self.handlers:
			handle.remove()

class GradCAM(_BaseWrapper):
	"""
	"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
	https://arxiv.org/pdf/1610.02391.pdf
	"""
	def __init__(self, model, candidate_layers=None):
		super(GradCAM, self).__init__(model)
		self.fmap_pool = {}
		self.grad_pool = {}
		self.candidate_layers = candidate_layers

		def save_fmaps(key):
			def forward_hook(module, input, output):
				self.fmap_pool[key] = output.detach()

			return forward_hook

		def save_grads(key):
			def backward_hook(module, grad_in, grad_out):
				self.grad_pool[key] = grad_out[0].detach()

			return backward_hook

		
		for name, module in self.model.named_modules():
			if self.candidate_layers is None or name in self.candidate_layers:
				self.handlers.append(module.register_forward_hook(save_fmaps(name)))
				self.handlers.append(module.register_backward_hook(save_grads(name)))

	def _find(self, pool, target_layer):
		if target_layer in pool.keys():
			return pool[target_layer]
		else:
			raise ValueError("Invalid layer name: {}".format(target_layer))

	def generate(self, target_layer):
		fmaps = self._find(self.fmap_pool, target_layer)
		grads = self._find(self.grad_pool, target_layer)
		weights = F.adaptive_avg_pool2d(grads, 1)

		gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
		gcam = F.relu(gcam)
		gcam = F.interpolate(
			gcam, self.image_shape, mode="bilinear", align_corners=False
		)

		B, C, H, W = gcam.shape
		gcam = gcam.view(B, -1)
		gcam -= gcam.min(dim=1, keepdim=True)[0]
		gcam /= gcam.max(dim=1, keepdim=True)[0]
		gcam = gcam.view(B, C, H, W)

		return gcam
