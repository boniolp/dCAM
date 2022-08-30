import os
import sys
from submit_generic import head


if __name__ == '__main__':
		
	dataset_names = [
		'AtrialFibrillation.pickle',
		'Libras.pickle',
		'BasicMotions.pickle',
		'RacketSports.pickle',
		'Epilepsy.pickle',
		'StandWalkJump.pickle',
		'UWaveGestureLibrary.pickle',
		'Handwriting.pickle',
		'NATOPS.pickle',
		'FingerMovements.pickle',
		'ArticularyWordRecognition.pickle',
		'HandMovementDirection.pickle',
		'Cricket.pickle',
		'LSST.pickle',
		'EthanolConcentration.pickle',
		'SelfRegulationSCP1.pickle',
		'SelfRegulationSCP2.pickle',
		'Heartbeat.pickle',
		'PhonemeSpectra.pickle',
		'EigenWorms.pickle',
		'MotorImagery.pickle',
		'PEMS-SF.pickle',
		'FaceDetection.pickle',
	]
	

	
	with open("submit_all.sh",'w') as f:
		f.write("#!/bin/bash \n")
	count = 0
	for name in dataset_names: 
			
		dataset_name = '../../../data/UCR_UEA/{}'.format(name)

		device='cuda'
				
		# Parameter in papers and for the results depicted in the paper
		#parameters = {
		#       'train_test_r': 0.80,
		#       'batch_size': 32,
		#       'nb_epoch': 1000,
		#       'nb_repeat_iteration': 10
		#}

		# Parameter to test the code
		parameters = {
			'train_test_r': 0.80,
			'batch_size': 8,
			'nb_epoch': 1000,
			'nb_repeat_iteration': 3
		}



		submit_name = 'auto_script/submit_{}.sh'.format(name.strip('.pickle'))
						
		with open("submit_all.sh",'a') as f:
			f.write("sbatch {}\n".format(submit_name))
			
		count += 1
		with open(submit_name,'w') as f:
			script_name = 'script_exp_dataset.py {} {} {} {} {}'.format(dataset_name,parameters['nb_epoch'],parameters['nb_repeat_iteration'], parameters['batch_size'], parameters['train_test_r'])
			name_job = 'UCR_{}'.format(name.strip('.pickle'))
			f.write(head.format(name_job,name_job,name_job,script_name))
