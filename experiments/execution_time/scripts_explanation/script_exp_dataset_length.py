import numpy as np
import pandas as pd
import os
from random import randint

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../../../src/models')
sys.path.append('../../../src/explanation')
from CNN_models import *
from RNN_models import *
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
def generate_list_instance(x,length=100,label=0):
	res = []
	for i in range(len(x)):
		if length > len(x[i]):
			tmp = []
			while length > len(tmp):
				tmp = tmp + list(x[i])
			if label==0:
				res.append(list(tmp[:length]))
			else:
				res.append(list(tmp[:length])[::-1])
		else:
			if label==0:
				res.append(list(x[i][:length]))
			else:
				res.append(list(x[i][:length])[::-1])
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



##### MODEL PREPROCESSING ######
# - Scripts related to models preparation and oarameter preprocessing

def gen_model(arg,original_length,original_dim,num_classes):
		
	#d-Baselines
	if arg == 'dcnn':
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

def process_dataset(dataset_name,train_test_r,batch_size,type_input='baseline',target_length=100):
		
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
		all_class_all.append(generate_list_instance(X.values[i],target_length,dict_label[y.values[i]]))
		all_label.append(dict_label[y.values[i]])


	original_length = len(all_class_all[0][0])
	num_classes = len(set(y.values))
	original_dim = len(all_class_all[0])
	nb_instance = len(all_class_all)

	all_class, all_class_test, label, label_test = train_test_split(all_class_all, all_label,
															stratify=all_label, 
															test_size=1-train_test_r,random_state=11081994)

   
	dict_dataset = {
		'all_class' : all_class_test,
		'label_test'  : label_test,
		'ts_length'    : original_length,
		'nb_classes'   : num_classes,
		'nb_dim'       : original_dim,
		'nb_instance'  : nb_instance,
	}

	return dict_dataset


##### MODEL EXEC ######
# - Scripts executing a given model on a given dataset
def exec_model(model_name,type_input,dataset_name,parameters,target_length):
		
	dict_dataset = process_dataset(
		dataset_name,
		parameters['train_test_r'],
		parameters['batch_size'],
		type_input,target_length)     

	all_time_epoch  = []
	for iteration in range(parameters['nb_repeat_iteration']):
			
		model_name_load = '../model/{}_{}_{}'.format(model_name,dataset_name.split('/')[-1].strip('.pickle'),target_length)
		model = torch.load(model_name_load).to(device)
		model = model.eval()
		count_instance = 0
		all_time = []
		for index_instance in range(len(dict_dataset['all_class'])):
			instance = dict_dataset['all_class'][index_instance]
			label_instance = dict_dataset['label_test'][index_instance]
			if count_instance == 10:
					break
			if label_instance == 1:
		
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
					start_time = time.time()
					explanation,nb_permutation_success = DCAM_m.run(instance=instance,
									nb_permutation=100,
									label_instance=label_instance)
					end_time = time.time()
					time_dcam = end_time - start_time
					all_time.append(time_dcam)
					count_instance += 1
				except:
					print('failed dcam')

				

		
		
	file_result = "../results_explanation/log_length/{}_{}_{}.txt".format(model_name,dataset_name.split('/')[-1].strip('.pickle'),target_length)

	with open(file_result ,"w") as f:
		f.write("{}-{}".format(np.mean(all_time),np.std(all_time)))



if __name__ == '__main__':

	print(sys.argv)

	dataset_name = sys.argv[3]
	model_name = sys.argv[1]
	type_input = sys.argv[2]


	parameters = {}
	parameters['train_test_r'] = float(sys.argv[7])
	parameters['batch_size'] = int(sys.argv[6])
	parameters['nb_epoch'] = int(sys.argv[4])
	parameters['nb_repeat_iteration'] = int(sys.argv[5])
	
	device = DEVICE
	for target_length in [10,20,50,100,200,500,1000,5000,10000]:
		exec_model(model_name,type_input,dataset_name,parameters,target_length)


