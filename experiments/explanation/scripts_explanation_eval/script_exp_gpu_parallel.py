import os
import sys
from submit_generic import head


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
		'synth_ShapesAll_type2_nbdim_100.pickle',
		'synth_starlight_type1_nbdim_10.pickle',
		'synth_starlight_type1_nbdim_20.pickle',
		'synth_starlight_type1_nbdim_40.pickle',
		'synth_starlight_type1_nbdim_60.pickle',
		'synth_starlight_type1_nbdim_100.pickle',
		'synth_starlight_type2_nbdim_10.pickle',
		'synth_starlight_type2_nbdim_20.pickle',
		'synth_starlight_type2_nbdim_40.pickle',
		'synth_starlight_type2_nbdim_60.pickle',
		'synth_starlight_type2_nbdim_100.pickle',
	]
	
	model_names = [
		('mtex','c'),
		('resnet','baseline'),
		('cresnet','c'),
		('dcnn','d'),
		('dresnet','d'),
		('dinception','d'),
	]

	try:
		with open("submit_all.sh",'w') as f:
			f.write("#!/bin/bash \n")
		count = 0
		for name in dataset_names: 
				
			dataset_name = '../../../data/synthetic/{}'.format(name)
			
			
			
			parameters = {
					'train_test_r': 0.80,
			}

			for model_name,type_input in model_names:
				submit_name = 'auto_script/submit_{}_{}.sh'.format(model_name,name.strip('.pickle'))
				
				with open("submit_all.sh",'a') as f:
					f.write("sbatch {}\n".format(submit_name))
					#f.write("sleep 5 \n")
				count += 1
				with open(submit_name,'w') as f:
					script_name = 'script_exp_dataset.py {} {} {} {}'.format(model_name,type_input,dataset_name, parameters['train_test_r'])
					name_job = 'synth_explanation_{}_{}'.format(model_name,name.strip('.pickle'))
					f.write(head.format(name_job,name_job,name_job,script_name))
	except KeyboardInterrupt:
			print('interuption...')
