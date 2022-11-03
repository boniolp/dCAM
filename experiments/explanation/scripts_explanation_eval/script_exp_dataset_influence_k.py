import numpy as np
import pandas as pd
import os
from random import randint
import random

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc,precision_recall_curve,roc_auc_score


import sys
sys.path.append('../../../src/models')
sys.path.append('../../../src/explanation')

from CNN_models import *
from RNN_models import *

from grad_cam_mtex import *
from CAM import *
from cCAM import *
from DCAM import *


from tqdm import tqdm
import pickle


# import device (cpu or cuda). 
#Default is cuda (for gpu server). 
#Please change file setting variabel DEVICE to change server type
from setting import DEVICE

##### DATA PREPROCESSING ######
# - Scripts related to data and input preprocessing

# Convert the UCR-UEA format into a list of list
def generate_list_instance(x):
	res = []
	for i in range(len(x)):
		res.append(list(x[i]))
	return np.array(res)

# Generate column-wised input for c-based models (i.e., cCNN, cResNet, and cInceptionTime)
def gen_col(instance):
	result = []
	for i in range(len(instance)):
			result.append(instance[i])
	return [result]

# Generate C-wised input for d-based models (i.e., dCNN, dResNet, and dInceptionTime)
def gen_cube(instance):
	result = []
	for i in range(len(instance)):
			result.append([instance[(i+j)%len(instance)] for j in range(len(instance))])
	return result 

def compute_AUC(dcam,gt):
	to_evaluate =[]
	gt_conc = []
	for i in range(len(dcam)):
		to_evaluate += list(dcam[i])
		gt_conc += list(gt[i])
	to_evaluate = (np.array(to_evaluate) - min(to_evaluate))/(max(to_evaluate) - min(to_evaluate))

	precision, recall, thresholds = precision_recall_curve(gt_conc,to_evaluate)
	auc_PR = auc(recall,precision)
	return auc_PR#roc_auc_score(gt_conc,to_evaluate)

##### MODEL PREPROCESSING ######
# - Scripts related to models preparation and oarameter preprocessing

def gen_model(arg,original_length,original_dim,num_classes):
		
		#Baselines
	if arg == 'mtex':
		modelarch = ConvNetMTEX(original_length,original_dim,1,num_classes).to(device)
		return ModelCNN(modelarch,device)
	elif arg == 'cnn':
		modelarch = ConvNet(original_length,original_dim,num_classes).to(device)
		return ModelCNN(modelarch,device)
	elif arg == 'resnet':
		modelarch = ResNetBaseline(original_dim,mid_channels=64,num_pred_classes=num_classes).to(device)
		return ModelCNN(modelarch,device)
	elif arg == 'inception':
		modelarch = InceptionModel(num_blocks=3, in_channels=original_dim, out_channels=32,
											bottleneck_channels=32, kernel_sizes=[10,20,40],
											use_residuals=True, num_pred_classes=num_classes).to(device)
		return ModelCNN(modelarch,device)

	#RNN Baselines
	elif arg == 'lstm':
		modelarch = LSTMClassifier(original_dim,original_length,128,3,num_classes,device).to(device)
		return ModelCNN(modelarch,device)
	elif arg == 'rnn':
		modelarch = RNNClassifier(original_dim,original_length,128,3,num_classes,device).to(device)
		return ModelCNN(modelarch,device)
	elif arg == 'gru':
		modelarch = GRUClassifier(original_dim,original_length,128,3,num_classes,device).to(device)
		return ModelCNN(modelarch,device)



	#c-Baselines
	elif arg == 'ccnn':
		modelarch = ConvNet2D(original_length,original_dim,1,num_classes).to(device)
		return ModelCNN(modelarch,device)
	elif arg == 'cresnet':
		modelarch = dResNetBaseline(1,mid_channels=128,num_pred_classes=num_classes).to(device)
		return ModelCNN(modelarch,device)
	elif arg == 'cinception':
		modelarch = dInceptionModel(num_blocks=3, in_channels=1, out_channels=64,
											bottleneck_channels=64, kernel_sizes=[10,20,40],
											use_residuals=True, num_pred_classes=num_classes).to(device)
		return ModelCNN(modelarch,device)

	#d-Baselines
	elif arg == 'dcnn':
		modelarch = ConvNet2D(original_length,original_dim,original_dim,num_classes).to(device)
		return ModelCNN(modelarch,device)
	elif arg == 'dresnet':
		modelarch = dResNetBaseline(original_dim,mid_channels=128,num_pred_classes=num_classes).to(device)
		return ModelCNN(modelarch,device)
	elif arg == 'dinception':
		modelarch = dInceptionModel(num_blocks=3, in_channels=original_dim, out_channels=64,
											bottleneck_channels=64, kernel_sizes=[10,20,40],
											use_residuals=True, num_pred_classes=num_classes).to(device)
		return ModelCNN(modelarch,device)


##### DATASET PREPROCESSING ######
# - Scripts generating the train and the test dataset and the relevant information from a specific dataset

def process_dataset(dataset_name,train_test_r,type_input='baseline'):
		
	with open(dataset_name,'rb') as f:
		X,y,X_gt = pickle.load(f)
			
	dict_label = {}
	count = 0
	for val in set(y.values):
		dict_label[val] = count
		count += 1

	all_class_all = []
	all_label = []
	for i in range(len(X)):
		all_class_all.append((generate_list_instance(X.values[i]),X_gt.values[i]))
		all_label.append(dict_label[y.values[i]])


	original_length = len(all_class_all[0][0][0])
	num_classes = len(set(y.values))
	original_dim = len(all_class_all[0][0])
	nb_instance = len(all_class_all)

	all_class, all_class_test, label, label_test = train_test_split(all_class_all, all_label,
																											stratify=all_label, 
																											test_size=1-train_test_r,random_state=11081994)

	all_class = [elem[0] for elem in all_class_test]
	all_class_gt = [elem[1] for elem in all_class_test]



	dict_dataset = {
		'all_class'     : all_class,
		'all_class_gt'  : all_class_gt,
		'label_test'    : label_test,
		'ts_length'     : original_length,
		'nb_classes'    : num_classes,
		'nb_dim'        : original_dim,
		'nb_instance'   : nb_instance,
	}

	return dict_dataset


##### EXEC MODEL ######
# - Scripts executing a given model on a given dataset
# - parameters is a dictionary containing the following items:
#     - train_test_r: the split ratio between the train and the test dataset 

def exec_model(model_name,type_input,dataset_name,parameters):
		
	dict_dataset = process_dataset(
		dataset_name,
		parameters['train_test_r'],
		type_input)     
	
	model_name_load='../models/{}_{}'.format(model_name,dataset_name.split('/')[-1].strip('.pickle'))
	model = torch.load(model_name_load).to(device)
	model = model.eval()
	all_score = []
	nb_success = []
	count_instance = 0    
	nb_perm_k_list = [1,2,3,4,5,6,7,8,9,10,20,40,60,80,100,150,200,400]    
	for index_instance in range(len(dict_dataset['all_class'])):
		instance = dict_dataset['all_class'][index_instance]
		label_instance = dict_dataset['label_test'][index_instance]
		if count_instance == 20:
			break
		if label_instance == 1:
			count_instance += 1
			if model_name == 'mtex':
				continue
			elif model_name == 'resnet':
				continue
			elif model_name == 'cresnet':
				continue 
			else:
				if model_name == 'dcnn':
					last_conv_layer = model._modules.get('layer3')
					fc_layer_name = model._modules.get('fc1')

				elif model_name == 'dresnet':
					last_conv_layer = model._modules['layers'][2]
					fc_layer_name = model._modules['final']

				elif model_name == 'dinception':
					last_conv_layer = model._modules['blocks'][2]
					fc_layer_name = model._modules['linear']

				DCAM_m = DCAM(model,
					device,
					last_conv_layer=last_conv_layer,
					fc_layer_name=fc_layer_name)

				try:
					dcam_list = DCAM_m.run_list_k(instance=instance,
						nb_permutation=nb_perm_k_list,
						label_instance=label_instance)
					tmp_score = []
					for dcam in dcam_list:
						gt = dict_dataset['all_class_gt'][index_instance]
						score = compute_AUC(dcam,gt)
						tmp_score.append(score)
					all_score.append(tmp_score)
					print(np.mean(all_score,0))                        
				except:
					print('failed dcam')
			

	file_result = "../results_parameters/log_k_exp/{}_{}.txt".format(model_name,dataset_name.split('/')[-1].strip('.pickle'))
	with open(file_result ,"w") as f:
		for dim,mean,std in zip(nb_perm_k_list,np.mean(all_score,0),np.std(all_score,0)):
			f.write("{}:{}-{}\n".format(dim,mean,std))
		

if __name__ == '__main__':

	print(sys.argv)

	dataset_name = sys.argv[3]
	model_name = sys.argv[1]
	type_input = sys.argv[2]


	parameters = {}
	parameters['train_test_r'] = float(sys.argv[4])
	
	
	device = DEVICE

	exec_model(model_name,type_input,dataset_name,parameters)
