import random
import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from torch import topk
from tqdm import tqdm

class SaveFeatures():
	features=None
	def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
	def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
	def remove(self): self.hook.remove()


class DCAM():
	def __init__(self,model,device,last_conv_layer='layer3',fc_layer_name='fc1'):
		
		self.device = device
		self.last_conv_layer = last_conv_layer
		self.fc_layer_name = fc_layer_name
		self.model = model




	def run(self,instance,nb_permutation,label_instance):
		
		all_permut,permut_success = self.__compute_permutations(instance, nb_permutation,label_instance)
		dcam =  self.__extract_dcam(self.__merge_permutation(all_permut))
		return dcam,permut_success

	
	def run_list_k(self,instance,nb_permutation,label_instance):
		k_max = nb_permutation[-1]
		all_permut,permut_success = self.__compute_permutations(instance, k_max, label_instance)
		all_dcams = []
		all_avg_mat = self.__merge_permutation_list_k(all_permut,nb_permutation)
		for mat in tqdm(all_avg_mat):
			dcam =  self.__extract_dcam(mat)
			all_dcams.append(dcam)
		return all_dcams


	

	# ================Private methods=====================   

	def __gen_cube_random(self,instance):
		result = []
		result_comb = []
		initial_comb = list(range(len(instance)))
		random.shuffle(initial_comb)
		for i in range(len(instance)):
			result.append([instance[initial_comb[(i+j)%len(instance)]] for j in range(len(instance))])
			result_comb.append([initial_comb[(i+j)%len(instance)] for j in range(len(instance))]) 
		
		return result,result_comb


	def __merge_permutation(self,all_matfull_list):
		full_mat_avg = np.zeros((len(all_matfull_list[0]),len(all_matfull_list[0][0]),len(all_matfull_list[0][0][0])))
		for i in range(len(all_matfull_list[0])):
			for j in range(len(all_matfull_list[0][0])):
				mean_line = np.array([np.mean([all_matfull_list[k][i][j][n] for k in range(len(all_matfull_list))]) for n in range(len(all_matfull_list[0][0][0]))])
				full_mat_avg[i][j] = mean_line
		return full_mat_avg


	def __merge_permutation_list_k(self,all_matfull_list,list_k):
		all_full_mat_avg = [np.zeros((len(all_matfull_list[0]),len(all_matfull_list[0][0]),len(all_matfull_list[0][0][0]))) for nb_k in range(len(list_k))]
		for i in tqdm(range(len(all_matfull_list[0]))):
			for j in range(len(all_matfull_list[0][0])):
				all_mean_line = [[] for nb_k in range(len(list_k))]
				tmp_mean_line_n = []
				for n in range(len(all_matfull_list[0][0][0])):
					tmp_mean_line_k = []
					for k in range(len(all_matfull_list)):
						tmp_mean_line_k.append(all_matfull_list[k][i][j][n])
					for nb_k in range(len(list_k)):
						all_mean_line[nb_k].append(np.mean(tmp_mean_line_k[:list_k[nb_k]]))
				for nb_k in range(len(list_k)):
					all_full_mat_avg[nb_k][i][j] = all_mean_line[nb_k]
		return all_full_mat_avg




	def __extract_dcam(self,full_mat_avg):
		return np.mean((full_mat_avg-np.mean(full_mat_avg,1))**2,1)*np.mean(np.mean(full_mat_avg,1),0)


	def __getCAM(self,feature_conv, weight_fc, class_idx):
		_,nch, nc, length = feature_conv.shape
		feature_conv_new = feature_conv
		cam = weight_fc[class_idx].dot(feature_conv_new.reshape((nch,nc*length)))
		cam = cam.reshape(nc,length)
		cam = (cam - np.min(cam))/(np.max(cam) - np.min(cam))
		return cam


	def __get_CAM_class(self,instance):
		original_dim = len(instance)
		original_length = len(instance[0][0])
		instance_to_try = Variable(
			torch.tensor(
				instance.reshape(
					(1,original_dim,original_dim,original_length))).float().to(self.device),
			requires_grad=True)
		final_layer  = self.last_conv_layer
		activated_features = SaveFeatures(final_layer)
		prediction = self.model(instance_to_try)
		pred_probabilities = F.softmax(prediction).data.squeeze()
		activated_features.remove()
		weight_softmax_params = list(self.fc_layer_name.parameters())
		weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
		
		class_idx = topk(pred_probabilities,1)[1].int()
		overlay = self.__getCAM(activated_features.features, weight_softmax, class_idx )
		
		return overlay,class_idx.item()



	def __compute_multidim_cam(self,instance,nb_dim,index_perm):
		acl,comb = self.__gen_cube_random(instance)
		overlay,pred_class = self.__get_CAM_class(np.array(acl))
		full_mat = np.zeros((nb_dim,nb_dim,len(overlay[0])))
		for i in range(nb_dim):
			for j in range(nb_dim):
				full_mat[comb[i][j]][i] = overlay[j]
		
		return overlay,full_mat,pred_class


	def __compute_permutations(self,instance, nb_permutation,label_instance):
		all_pred_class = []
		all_matfull_list = []

		final_mat = np.zeros((len(instance),len(instance)))
		for k in tqdm(range(0,nb_permutation)):
			_,fmat,class_pred = self.__compute_multidim_cam(instance,len(instance),k)
			if class_pred == label_instance:
				all_matfull_list.append(fmat)
			all_pred_class.append(class_pred)
		if np.std(all_pred_class) == 0:
			#verbose
			#print("[INFO]: No misclassification for all permutations")
			return all_matfull_list,nb_permutation
		else:
			#verbose
			#print("[WARNING]: misclassification for some permutations:")
			#print("|--> Please note that every misclassified permutations will not be taken into account")
			#print("|--> The total number of permutation used to compute DCAM is lower than the number given as parameter")
			#print("|--> Number correctly classified permutations: {}".format(all_pred_class.count(label_instance)))
			#print("|--> Percentage of correctly classified permutations: {}".format(float(all_pred_class.count(label_instance))/float(len(all_pred_class))))
			return all_matfull_list,float(all_pred_class.count(label_instance))
	
	
	
