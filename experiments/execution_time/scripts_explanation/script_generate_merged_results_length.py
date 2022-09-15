import numpy as np
import pandas as pd
import os
import sys

import pickle

def generate_output_file(dataset_names,model_names,lengths):
	df_results = pd.DataFrame(
			index=[length for length in lengths],
			columns=[name[0] for name in model_names])
	for file in os.listdir('../results_explanation/log_length/'):
		if '.txt' in file:
			model_name = file.split('_')[0]
			length = int(file.split('_')[-1].replace('.txt',''))
		
			with open('../results_explanation/log_length/'+file ,"r") as f:
				for line in f:
					val = line.rstrip()
			df_results.at[length,model_name] = float(val.split('-')[0])

	df_results.to_csv('../results_explanation/merged_results_length.csv')



if __name__ == '__main__':
	

	dataset_names = [
		'synth_ShapesAll_type1_nbdim_10.pickle',
	]
	
	lengths = [10,20,50,100,200,500,1000,5000,10000]

	model_names = [
		('dcnn','d'),
		('dresnet','d'),
		('dinception','d'),
	]
	generate_output_file(dataset_names,model_names,lengths)



			







