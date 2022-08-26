import random
import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from torch import topk


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()


class cCAM():
    def __init__(self,model,device,last_conv_layer='layer3',fc_layer_name='fc1'):
        
        self.device = device
        self.last_conv_layer = last_conv_layer
        self.fc_layer_name = fc_layer_name
        self.model = model


    def run(self,instance,label_instance=None):
        cam,label_pred =  self.__get_CAM_class(np.array(instance))
        if (label_instance is not None) and (label_pred != label_instance):
            return None
            #verbose
            #print("[WARNING] expected classification as class {} but got class {}".format(label_instance,label_pred))
            #print("[WARNING] The Class activation map is for class {}".format(label_instance,label_pred))
        return cam

    

    # ================Private methods=====================   

    def __getCAM(self,feature_conv, weight_fc, class_idx):
        _, nch,nc, length = feature_conv.shape
        feature_conv_new = feature_conv
        cam = weight_fc[class_idx].dot(feature_conv_new.reshape((nch,nc*length)))
        cam = cam.reshape(nc,length)
        
        return cam


    def __get_CAM_class(self,instance):
        original_dim = len(instance[0])
        original_length = len(instance[0][0])
        instance_to_try = Variable(
            torch.tensor(
                instance.reshape(
                    (1,1,original_dim,original_length))).float().to(self.device),
            requires_grad=True)
        final_layer = self.last_conv_layer
        activated_features = SaveFeatures(final_layer)
        prediction = self.model(instance_to_try)
        pred_probabilities = F.softmax(prediction).data.squeeze()
        activated_features.remove()
        weight_softmax_params = list(self.fc_layer_name.parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
        
        class_idx = topk(pred_probabilities,1)[1].int()
        overlay = self.__getCAM(activated_features.features, weight_softmax, class_idx )
        
        return overlay,class_idx.item()
