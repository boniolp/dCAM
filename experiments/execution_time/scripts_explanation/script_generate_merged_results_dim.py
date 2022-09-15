import numpy as np
import pandas as pd
import os
import sys

import pickle

def generate_output_file(dataset_names,model_names,list_dim):
	df_results = pd.DataFrame(
			index=[dim for dim in list_dim],
			columns=[name[0] for name in model_names])
	for file in os.listdir('../results_explanation/log_dim/'):
		if '.txt' in file:
			model_name = file.split('_')[0]
			dim = int(file.split('_')[-1].replace('.txt',''))
		
			with open('../results_explanation/log_dim/'+file ,"r") as f:
				for line in f:
					val = line.rstrip()
			df_results.at[dim,model_name] = float(val.split('-')[0])

	df_results.to_csv('../results_explanation/merged_results_dim.csv')



if __name__ == '__main__':
	

	dataset_names = [
		'synth_ShapesAll_type1_nbdim_10.pickle',
	]
	
	list_dim = [10,20,40,60,100]

	model_names = [
		('dcnn','d'),
		('dresnet','d'),
		('dinception','d'),
	]
	generate_output_file(dataset_names,model_names,list_dim)



			







