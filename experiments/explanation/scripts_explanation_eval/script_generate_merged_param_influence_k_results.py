import numpy as np
import pandas as pd
import os
import sys

import pickle

def generate_output_file(dataset_names,model_names):
        df_results = pd.DataFrame(
                        index=[dataset_name.split('/')[-1].strip('.pickle') for dataset_name in dataset_names],
                        columns=[name[0] for name in model_names])
        for file in os.listdir('../results_parameters/log_k_exp/'):
                if '.txt' in file:
                        model_name = file.split('_')[0]
                        dataset_name  = file.replace(model_name+'_','').replace('.txt','')
                        list_values = ''
                        with open('../results_parameters/log_k_exp/'+file ,"r") as f:
                                for line in f:
                                        val = line.rstrip()
                                        list_values += str(float(val.split(':')[1].split('-')[0]))
                                        list_values += ';'
                        df_results.at[dataset_name,model_name] = list_values
        df_results.to_csv('../results_parameters/merged_results_influence_k.csv')



if __name__ == '__main__':
        

        dataset_names = [
                'synth_ShapesAll_type1_nbdim_10.pickle',
                'synth_ShapesAll_type1_nbdim_20.pickle',
                'synth_ShapesAll_type1_nbdim_40.pickle',
                'synth_ShapesAll_type1_nbdim_60.pickle',
                'synth_ShapesAll_type1_nbdim_100.pickle',
                'synth_ShapesAll_type2_nbdim_10.pickle',
                'synth_ShapesAll_type2_nbdim_20.pickle',
                'synth_ShapesAll_type2_nbdim_40.pickle',
                'synth_ShapesAll_type2_nbdim_60.pickle',
        ]
        
        model_names = [
                ('dcnn','d'),
                ('dresnet','d'),
                ('dinception','d'),
        ]

        generate_output_file(dataset_names,model_names)



                        







