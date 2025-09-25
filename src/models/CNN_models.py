
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
import time


class TSDataset(data.Dataset):
		def __init__(self,x_train,labels):
				self.samples = x_train
				self.labels = labels

		def __len__(self):
				return len(self.samples)

		def __getitem__(self,idx):
				return self.samples[idx],self.labels[idx]


class ModelCNN():
		def __init__(self,
				model,
				device="cpu",
				criterion=nn.CrossEntropyLoss(),
				n_epochs_stop=300,
				learning_rate=0.00001):
				
				self.model = model
				self.n_epochs_stop = n_epochs_stop
				self.criterion = criterion
				self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
				self.device = device
				self.training_time_epoch = 0
				self.epoch_best = 0

		def __test(self,dataloader):
				mean_loss = []
				mean_accuracy = []
				total_sample = []
				
				with torch.no_grad():
						for i,batch_data in enumerate(dataloader):
								self.model.eval()
								ts, label = batch_data
								img = Variable(ts.float()).to(self.device)
								v_label = Variable(label.float()).to(self.device)
								# ===================forward=====================
								output = self.model(img.float()).to(self.device)
								
								loss = self.criterion(output.float(), v_label.long())
								
								# ================eval on test===================
								total = label.size(0)
								_, predicted = torch.max(output.data, 1)
								correct = (predicted.to(self.device) == label.to(self.device)).sum().item()
								
								mean_loss.append(loss.item())
								mean_accuracy.append(correct)
								total_sample.append(total)
				
				return mean_loss,mean_accuracy,total_sample

		def train(self,num_epochs,dataloader_cl1,dataloader_cl1_test,model_name='model',verbose=True):
				epochs_no_improve = 0
				min_val_loss = np.inf
				loss_train_history = []
				loss_test_history = []
				accuracy_test_history = []
				
				for epoch in range(num_epochs):
						mean_loss_train = []
						mean_accuracy_train = []
						total_sample_train = []
		
						time_start = time.time()
						for i,batch_data_train in enumerate(dataloader_cl1):
								self.model.train()
								ts_train, label_train = batch_data_train
								
								img_train = Variable(ts_train.float()).to(self.device)
								v_label_train = Variable(label_train.float()).to(self.device)
								
								# ===================forward=====================
								self.optimizer.zero_grad()
								output_train = self.model(img_train.float()).to(self.device)
								
								# ===================backward====================
								loss_train = self.criterion(output_train.float(), v_label_train.long())
								loss_train.backward()
								self.optimizer.step()
								# ================eval on train==================
								total_train = label_train.size(0)
								_, predicted_train = torch.max(output_train.data, 1)
								correct_train = (predicted_train.to(self.device) == label_train.to(self.device)).sum().item()
								mean_loss_train.append(loss_train.item())
								mean_accuracy_train.append(correct_train)
								total_sample_train.append(total_train)
						time_end = time.time()
						self.training_time_epoch = time_end - time_start
						# ==================eval on test=====================
						mean_loss_test,mean_accuracy_test,total_sample_test = self.__test(dataloader_cl1_test)

						# ====================verbose========================
						if (epoch % 10 == 0) and verbose:
								print('Epoch [{}/{}], Loss Train: {:.4f},Loss Test: {:.4f}, Accuracy Train: {:.2f}%, Accuracy Test: {:.2f}%'
										.format(epoch + 1, 
														num_epochs,
														np.mean(mean_loss_train),
														np.mean(mean_loss_test),
														(np.sum(mean_accuracy_train)/np.sum(total_sample_train)) * 100,
														(np.sum(mean_accuracy_test)/np.sum(total_sample_test)) * 100))

						# ======================log==========================
						loss_test_history.append(np.mean(mean_loss_test))
						loss_train_history.append(np.mean(mean_loss_train))
						accuracy_test_history.append(np.sum(mean_accuracy_test)/np.sum(total_sample_test))
						self.loss_test_history = loss_test_history
						self.loss_train_history = loss_train_history
						self.accuracy_test_history = accuracy_test_history
						# ================early stopping=====================
						if epoch == 3:
								min_val_loss = np.sum(mean_loss_test)

						if np.sum(mean_loss_test) < min_val_loss:
								if model_name is not None:
									torch.save(self.model, model_name)
								self.epoch_best = epoch
								epochs_no_improve = 0
								min_val_loss = np.sum(mean_loss_test)

						else:
								epochs_no_improve += 1
								if epochs_no_improve == self.n_epochs_stop:
										if model_name is not None:
											self.model = torch.load(model_name)
										break
						if (np.mean(mean_loss_test) < 0.05) and (np.sum(mean_accuracy_test)/np.sum(total_sample_test) == 1.0):
							break
				

# Model used for CNN baseline
class ConvNet(nn.Module):
		def __init__(self,original_length,original_dim,num_classes=10):
				super(ConvNet, self).__init__()
				
				
				self.num_class = num_classes

				self.kernel_size = 3
				self.padding = 1

				self.layer1 = nn.Sequential(
						nn.Conv1d(original_dim, 64, kernel_size=self.kernel_size, padding=self.padding),
						nn.BatchNorm1d(64),
						nn.ReLU(),
						)
						
				self.layer2 = nn.Sequential(
						nn.Conv1d(64, 128, kernel_size=self.kernel_size, padding=self.padding),
						nn.BatchNorm1d(128),
						nn.ReLU(),
						)
				
				self.layer21 = nn.Sequential(
						nn.Conv1d(128, 256, kernel_size=self.kernel_size, padding=self.padding),
						nn.BatchNorm1d(256),
						nn.ReLU(),
						)

				self.layer22 = nn.Sequential(
						nn.Conv1d(256, 256, kernel_size=self.kernel_size, padding=self.padding),
						nn.BatchNorm1d(256),
						nn.ReLU(),
						)

				self.layer23 = nn.Sequential(
						nn.Conv1d(256, 256, kernel_size=self.kernel_size, padding=self.padding),
						nn.BatchNorm1d(256),
						nn.ReLU(),
						)

				self.layer3 = nn.Sequential(
						nn.Conv1d(256, 256, kernel_size=self.kernel_size, padding=self.padding),
						nn.ReLU(),
						)
						
				self.GAP = nn.AvgPool1d(original_length)
				
				self.fc1 = nn.Sequential(nn.Linear(256,num_classes))
				
				
				
		def forward(self, x):
				out = self.layer1(x)
				out = self.layer2(out)
				out = self.layer21(out)
				out = self.layer22(out)
				out = self.layer3(out)
				
				out = self.GAP(out)
				out = out.reshape(out.size(0), -1)
				out = self.fc1(out)
				
				return out

# Model used for cCNN baseline and dCNN
class ConvNet2D(nn.Module):
		def __init__(self,original_length,original_dim,nb_channel,num_classes=10):
				super(ConvNet2D, self).__init__()
				
				self.kernel_size = (1,3)
				self.padding = (0,1)
				self.num_class = num_classes
				
				self.layer1 = nn.Sequential(
						nn.Conv2d(nb_channel, 128, kernel_size=self.kernel_size, padding=self.padding),
						nn.BatchNorm2d(128),
						nn.ReLU(),
						)
						
				self.layer2 = nn.Sequential(
						nn.Conv2d(128, 128, kernel_size = self.kernel_size, padding=self.padding),
						nn.BatchNorm2d(128),        
						nn.ReLU(),
						)
				
				self.layer21 = nn.Sequential(
						nn.Conv2d(128, 256, kernel_size = self.kernel_size, padding=self.padding),
						nn.BatchNorm2d(256),
						nn.ReLU(),
						)
				
				self.layer22 = nn.Sequential(
						nn.Conv2d(256, 256, kernel_size = self.kernel_size, padding=self.padding),
						nn.BatchNorm2d(256),
						nn.ReLU(),
						)
				self.layer23 = nn.Sequential(
						nn.Conv2d(256, 256, kernel_size = self.kernel_size, padding=self.padding),
						nn.BatchNorm2d(256),
						nn.ReLU(),
						)
				
				
				self.layer3 = nn.Sequential(
						nn.Conv2d(256, 256, kernel_size=self.kernel_size, padding=self.padding),
						nn.ReLU(),
						)
						
						
				self.GAP = nn.AvgPool2d(kernel_size=(original_dim,original_length))
				self.fc1 = nn.Sequential(nn.Linear(256,num_classes))
				
				
		
		def forward(self, x):
				out = self.layer1(x)
				out = self.layer2(out)
				out = self.layer21(out)
				out = self.layer22(out)
				#out = self.layer23(out)
				out = self.layer3(out)
				
				out = self.GAP(out)
				out = out.reshape(out.size(0), -1)
				out = self.fc1(out)
				return out



class ConvNetMTEX(nn.Module):
		def __init__(self,original_length,original_dim,nb_channel,num_classes=10):
				super(ConvNetMTEX, self).__init__()
				
				self.kernel_size = (1,3)
				self.padding = (0,1)
				self.num_class = num_classes
				
				self.layer1 = nn.Sequential(
						nn.Conv2d(1, 64, kernel_size=(1,8), padding=(0,3), stride=(1,2)),
						nn.BatchNorm2d(64),
						nn.ReLU(),
						)
						
				self.layer2 = nn.Sequential(
						nn.Conv2d(64, 128, kernel_size = (1,6), padding=(0,2), stride=(1,2)),
						nn.BatchNorm2d(128),        
						nn.ReLU(),
						)
				
				self.layer21 = nn.Sequential(
						nn.Conv2d(128, 1, kernel_size = (1,1)),
						nn.BatchNorm2d(1),
						nn.ReLU(),
						)
				
				self.layer22 = nn.Sequential(
						nn.Conv2d(1, 128, kernel_size = (original_dim,4), padding=(0,1),stride=(1,2)),
						nn.BatchNorm2d(128),
						nn.ReLU(),
						)
						
				self.fc1 = nn.Sequential(nn.Linear(128*(original_length//8),num_classes))
				self.layerflatten = nn.Sequential(nn.Flatten())
				
		
		def forward(self, x):
				out = self.layer1(x)
				out = self.layer2(out)
				out = self.layer21(out)
				out = self.layer22(out)
				out = self.layerflatten(out)
				out = out.reshape(out.size(0), -1)
				out = self.fc1(out)
				return out


########################################################################################
#SOURCE: https://github.com/TheMrGhostman/InceptionTime-Pytorch/blob/master/inception.py
########################################################################################
class InceptionModel(nn.Module):
	
	def __init__(self, num_blocks, in_channels, out_channels,
				 bottleneck_channels, kernel_sizes,
				 use_residuals = 'default',
				 num_pred_classes = 1
				 ):
		super().__init__()

		
		self.input_args = {
			'num_blocks': num_blocks,
			'in_channels': in_channels,
			'out_channels': out_channels,
			'bottleneck_channels': bottleneck_channels,
			'kernel_sizes': kernel_sizes,
			'use_residuals': use_residuals,
			'num_pred_classes': num_pred_classes
		}

		channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
																		  num_blocks))
		bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
																	 num_blocks))
		kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
		if use_residuals == 'default':
			use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
		use_residuals = cast(List[bool], self._expand_to_blocks(
			cast(Union[bool, List[bool]], use_residuals), num_blocks)
		)

		self.blocks = nn.Sequential(*[
			InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
						   residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
						   kernel_size=kernel_sizes[i]) for i in range(num_blocks)
		])

		self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)

	@staticmethod
	def _expand_to_blocks(value,
						  num_blocks):
		if isinstance(value, list):
			assert len(value) == num_blocks
		else:
			value = [value] * num_blocks
		return value

	def forward(self, x):
		x = self.blocks(x).mean(dim=-1)
		return self.linear(x)


class InceptionBlock(nn.Module):
	
	def __init__(self, in_channels, out_channels,
				 residual, stride = 1, bottleneck_channels = 32,
				 kernel_size = 41):
		super().__init__()

		self.use_bottleneck = bottleneck_channels > 0
		if self.use_bottleneck:
			self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
												kernel_size=1, bias=False)
		kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
		start_channels = bottleneck_channels if self.use_bottleneck else in_channels
		channels = [start_channels] + [out_channels] * 3
		self.conv_layers = nn.Sequential(*[
			Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
							  kernel_size=kernel_size_s[i], stride=stride, bias=False)
			for i in range(len(kernel_size_s))
		])

		self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
		self.relu = nn.ReLU()

		self.use_residual = residual
		if residual:
			self.residual = nn.Sequential(*[
				Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
								  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm1d(out_channels),
				nn.ReLU()
			])

	def forward(self, x):
		org_x = x
		if self.use_bottleneck:
			x = self.bottleneck(x)
		x = self.conv_layers(x)

		if self.use_residual:
			x = x + self.residual(org_x)
		return x



########################################################################################
#SOURCE: https://github.com/okrasolar/pytorch-timeseries/blob/master/
########################################################################################


class Conv1dSamePadding(nn.Conv1d):
		def forward(self, input):
				return conv1d_same_padding(input, self.weight, self.bias, self.stride,
																   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
		kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
		l_out = l_in = input.size(2)
		padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
		if padding % 2 != 0:
				input = F.pad(input, [0, 1])

		return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
										padding=padding // 2,
										dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

		def __init__(self, in_channels, out_channels, kernel_size,
								 stride):
				super().__init__()

				self.layers = nn.Sequential(
						Conv1dSamePadding(in_channels=in_channels,
														  out_channels=out_channels,
														  kernel_size=kernel_size,
														  stride=stride),
						nn.BatchNorm1d(num_features=out_channels),
						nn.ReLU(),
				)

		def forward(self, x):

				return self.layers(x)


class ResNetBaseline(nn.Module):
		
		def __init__(self, in_channels, mid_channels = 64,
								 num_pred_classes = 1):
				super().__init__()

				
				self.input_args = {
						'in_channels': in_channels,
						'num_pred_classes': num_pred_classes
				}

				self.layers = nn.Sequential(*[
						ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
						ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
						ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

				])
				self.final = nn.Linear(mid_channels * 2, num_pred_classes)

		def forward(self, x):
				x = self.layers(x)
				return self.final(x.mean(dim=-1))


class ResNetBlock(nn.Module):

		def __init__(self, in_channels, out_channels):
				super().__init__()

				channels = [in_channels, out_channels, out_channels, out_channels]
				kernel_sizes = [8, 5, 3]

				self.layers = nn.Sequential(*[
						ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
										  kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
				])

				self.match_channels = False
				if in_channels != out_channels:
						self.match_channels = True
						self.residual = nn.Sequential(*[
								Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
																  kernel_size=1, stride=1),
								nn.BatchNorm1d(num_features=out_channels)
						])

		def forward(self, x):

				if self.match_channels:
						return self.layers(x) + self.residual(x)
				return self.layers(x)
		




#########################################################################################
################################################ d-ResNet ###############################
#########################################################################################


class Conv2dSamePadding(nn.Conv2d):
		def forward(self, input):
				return conv2d_same_padding(input, self.weight, self.bias, self.stride,
																   self.dilation, self.groups)


def conv2d_same_padding(input, weight, bias, stride, dilation, groups):
		kernel, dilation, stride = weight.size(3), dilation[0], stride[0]
		l_out = l_in = input.size(3)
		padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
		if padding % 2 != 0:
				input = F.pad(input, [0, 1])

		return F.conv2d(input=input, weight=weight, bias=bias, stride=stride,
										padding=(0,padding // 2),
										dilation=dilation, groups=groups)


class ConvBlock2d(nn.Module):

		def __init__(self, in_channels, out_channels, kernel_size,
								 stride):
				super().__init__()

				self.layers = nn.Sequential(
						Conv2dSamePadding(in_channels=in_channels,
														  out_channels=out_channels,
														  kernel_size=kernel_size,
														  stride=stride),
						nn.BatchNorm2d(num_features=out_channels),
						nn.ReLU(),
				)

		def forward(self, x):
				return self.layers(x)


class dResNetBaseline(nn.Module):
		
		def __init__(self, in_channels, mid_channels = 64,
								 num_pred_classes = 1):
				super().__init__()

				self.input_args = {
						'in_channels': in_channels,
						'num_pred_classes': num_pred_classes
				}

				self.layers = nn.Sequential(*[
						ResNetBlock2d(in_channels=in_channels, out_channels=mid_channels),
						ResNetBlock2d(in_channels=mid_channels, out_channels=mid_channels * 2),
						ResNetBlock2d(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

				])
				self.final = nn.Linear(mid_channels * 2, num_pred_classes)

		def forward(self, x):
				x = self.layers(x)
				return self.final(x.mean(dim=-1).mean(dim=-1))


class ResNetBlock2d(nn.Module):

		def __init__(self, in_channels, out_channels):
				super().__init__()

				channels = [in_channels, out_channels, out_channels, out_channels]
				kernel_sizes = [(1,8), (1,5), (1,3)]

				self.layers = nn.Sequential(*[
						ConvBlock2d(in_channels=channels[i], out_channels=channels[i + 1],
										  kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
				])

				self.match_channels = False
				if in_channels != out_channels:
						self.match_channels = True
						self.residual = nn.Sequential(*[
								Conv2dSamePadding(in_channels=in_channels, out_channels=out_channels,
																  kernel_size=(1,1), stride=1),
								nn.BatchNorm2d(num_features=out_channels)
						])

		def forward(self, x):

				if self.match_channels:
						return self.layers(x) + self.residual(x)
				return self.layers(x)
		




#########################################################################################
################################################ d-inception ############################
#########################################################################################


class dInceptionModel(nn.Module):

	def __init__(self, num_blocks, in_channels, out_channels,
				 bottleneck_channels, kernel_sizes,
				 use_residuals = 'default',
				 num_pred_classes = 1
				 ):
		super().__init__()

		
		self.input_args = {
			'num_blocks': num_blocks,
			'in_channels': in_channels,
			'out_channels': out_channels,
			'bottleneck_channels': bottleneck_channels,
			'kernel_sizes': kernel_sizes,
			'use_residuals': use_residuals,
			'num_pred_classes': num_pred_classes
		}

		channels = [in_channels] + cast(List[int], self._expand_to_blocks(out_channels,
																		  num_blocks))
		bottleneck_channels = cast(List[int], self._expand_to_blocks(bottleneck_channels,
																	 num_blocks))
		kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
		if use_residuals == 'default':
			use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
		use_residuals = cast(List[bool], self._expand_to_blocks(
			cast(Union[bool, List[bool]], use_residuals), num_blocks)
		)

		self.blocks = nn.Sequential(*[
			InceptionBlock2d(in_channels=channels[i], out_channels=channels[i + 1],
						   residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
						   kernel_size=kernel_sizes[i]) for i in range(num_blocks)
		])

		self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)

	@staticmethod
	def _expand_to_blocks(value,
						  num_blocks):
		if isinstance(value, list):
			assert len(value) == num_blocks
		else:
			value = [value] * num_blocks
		return value

	def forward(self, x):
		x = self.blocks(x).mean(dim=-1).mean(dim=-1)
		return self.linear(x)


class InceptionBlock2d(nn.Module):
	
	def __init__(self, in_channels, out_channels,
				 residual, stride = 1, bottleneck_channels = 32,
				 kernel_size = 41):
		super().__init__()

		self.use_bottleneck = bottleneck_channels > 0
		if self.use_bottleneck:
			self.bottleneck = Conv2dSamePadding(in_channels, bottleneck_channels,
												kernel_size=1, bias=False)
		kernel_size_s = [(1,kernel_size // (2 ** i)) for i in range(3)]
		start_channels = bottleneck_channels if self.use_bottleneck else in_channels
		channels = [start_channels] + [out_channels] * 3
		self.conv_layers = nn.Sequential(*[
			Conv2dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
							  kernel_size=kernel_size_s[i], stride=stride, bias=False)
			for i in range(len(kernel_size_s))
		])

		self.batchnorm = nn.BatchNorm2d(num_features=channels[-1])
		self.relu = nn.ReLU()

		self.use_residual = residual
		if residual:
			self.residual = nn.Sequential(*[
				Conv2dSamePadding(in_channels=in_channels, out_channels=out_channels,
								  kernel_size=(1,1), stride=stride, bias=False),
				nn.BatchNorm2d(out_channels),
				nn.ReLU()
			])

	def forward(self, x):
		org_x = x
		if self.use_bottleneck:
			x = self.bottleneck(x)
		x = self.conv_layers(x)

		if self.use_residual:
			x = x + self.residual(org_x)
		return x




