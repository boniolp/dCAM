import numpy as np
import pandas as pd
import os
from random import randint
from idr_pytools import gpu_jobs_submitter, display_slurm_queue, search_log 

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../../../src/models')

from CNN_models import *
from RNN_models import *

from tqdm import tqdm
import pickle


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

def process_dataset(dataset_name,train_test_r,batch_size,type_input='baseline'):
        
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
                all_class_all.append(generate_list_instance(X.values[i]))
                all_label.append(dict_label[y.values[i]])


        original_length = len(all_class_all[0][0])
        num_classes = len(set(y.values))
        original_dim = len(all_class_all[0])
        nb_instance = len(all_class_all)

        all_class, all_class_test, label, label_test = train_test_split(all_class_all, all_label,
                                                                                                                stratify=all_label, 
                                                                                                                test_size=1-train_test_r,random_state=11081994)

        #1D-based models in which all dimensions are stored in channels
        if type_input == 'baseline':
                #Generate train dataloader
                dataset_mat = TSDataset(all_class,label)
                dataloader_cl1 = data.DataLoader(dataset_mat, batch_size=batch_size, shuffle=True)
                
                #Generate test dataloader
                dataset_mat_test = TSDataset(all_class_test,label_test)
                dataloader_cl1_test = data.DataLoader(dataset_mat_test, batch_size=1, shuffle=True)

        #2D-based models in which all dimensions are stored in different columns with only one channel.
        elif type_input == 'c':
                x = np.array([gen_col(acl) for acl in all_class])
                dataset_mat = TSDataset(x,label)
                dataloader_cl1 = data.DataLoader(dataset_mat, batch_size=batch_size, shuffle=True)

                x = np.array([gen_col(acl) for acl in all_class_test])
                dataset_mat_test = TSDataset(x,label_test)
                dataloader_cl1_test = data.DataLoader(dataset_mat_test, batch_size=1, shuffle=True)

        #2D-based models in which several permutations of all pdimensions are stored in different columns.
        elif type_input == 'd':
                x = np.array([gen_cube(acl) for acl in all_class])
                dataset_mat = TSDataset(x,label)
                dataloader_cl1 = data.DataLoader(dataset_mat, batch_size=batch_size, shuffle=True)

                x = np.array([gen_cube(acl) for acl in all_class_test])
                dataset_mat_test = TSDataset(x,label_test)
                dataloader_cl1_test = data.DataLoader(dataset_mat_test, batch_size=1, shuffle=True)

        
        dict_dataset = {
                'train_loader' : dataloader_cl1,
                'test_loader'  : dataloader_cl1_test,
                'ts_length'    : original_length,
                'nb_classes'   : num_classes,
                'nb_dim'       : original_dim,
                'nb_instance'  : nb_instance,
        }

        return dict_dataset


##### DATASET PREPROCESSING ######
# - Scripts executing a given model on a given dataset
# - parameters is a dictionary containing the following items:
#     - train_test_r: the split ratio between the train and the test dataset 
#     - batch_size: batch size
#     - nb_epoch: number of epoches
#     - nb_repeat_iteration: number of time the training is repeated.
#     - The average over the differnt iteration is returned.

def exec_model(model_name,type_input,dataset_name,parameters):
        
        dict_dataset = process_dataset(
                dataset_name,
                parameters['train_test_r'],
                parameters['batch_size'],
                type_input)     

        all_acc  = []
        for iteration in range(parameters['nb_repeat_iteration']):
                
                model = gen_model(
                        model_name,
                        dict_dataset['ts_length'],
                        dict_dataset['nb_dim'],
                        dict_dataset['nb_classes'])
                
                model.train(
                        num_epochs=parameters['nb_epoch'],
                        dataloader_cl1=dict_dataset['train_loader'],
                        dataloader_cl1_test=dict_dataset['test_loader'],
                        model_name='../models/{}_{}'.format(model_name,dataset_name.split('/')[-1].strip('.pickle')),
                        verbose=True)
                
                # Store the best accuracy on the test dataset and delete the model
                all_acc.append(max(model.accuracy_test_history))
                del model
                # Uncomment this line of used with GPU
                #torch.cuda.empty_cache()

        
        
        file_result = "../results_classification/log/{}_{}.txt".format(model_name,dataset_name.split('/')[-1].strip('.pickle'))

        with open(file_result ,"w") as f:
                f.write("{}-{}".format(np.mean(all_acc),np.std(all_acc)))



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
        
        device = 'cuda'

        exec_model(model_name,type_input,dataset_name,parameters)


