import numpy as np
import pandas as pd
import os
import sys

import pickle


def generate_output_file(dataset_names,model_names):
        df_results = pd.DataFrame(
                        index=[dataset_name.split('/')[-1].strip('.pickle') for dataset_name in dataset_names],
                        columns=[name[0] for name in model_names])
        for file in os.listdir('../results/log/'):
                if '.txt' in file:
                        model_name = file.split('_')[0]
                        dataset_name  = file.replace(model_name+'_','').replace('.txt','')
                        with open('../results/log/'+file ,"r") as f:
                                for line in f:
                                        val = line.rstrip()
                        df_results.at[dataset_name,model_name] = float(val.split('-')[0])

        df_results.to_csv('../results/merged_results_classification.csv')


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
        
        model_names = [
                ('mtex','c'),
                ('cnn','baseline'),
                ('resnet','baseline'),
                ('inception','baseline'),
                ('lstm','baseline'),
                ('rnn','baseline'),
                ('gru','baseline'),
                ('ccnn','c'),
                ('cresnet','c'),
                ('cinception','c'),
                ('dcnn','d'),
                ('dresnet','d'),
                ('dinception','d'),
        ]

        generate_output_file(dataset_names,model_names)



                        







