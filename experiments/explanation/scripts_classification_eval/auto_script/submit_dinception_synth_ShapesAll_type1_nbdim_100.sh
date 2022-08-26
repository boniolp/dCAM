#!/bin/bash
#SBATCH --job-name=synth_dinception_synth_ShapesAll_type1_nbdim_100        					# name of job
#SBATCH --ntasks=1                  			# total number of processes (= number of GPUs here)
#SBATCH --gres=gpu:1                			# number of GPUs (1/4 of GPUs)
#SBATCH --cpus-per-task=10          			# number of cores per task (1/4 of the 4-GPUs node)
#SBATCH --hint=nomultithread         			# hyperthreading is deactivated
#SBATCH --time=20:00:00              			# maximum execution time requested (HH:MM:SS)
#SBATCH --output=auto_script/output_log/synth_dinception_synth_ShapesAll_type1_nbdim_100.out  # name of output file
#SBATCH --error=auto_script/output_log/synth_dinception_synth_ShapesAll_type1_nbdim_100.out   # name of error file (here, in common with the output file)


# cleans out the modules loaded in interactive and inherited by default 
module purge
 
# loading of modules
module load pytorch-gpu/py3/1.7.0
 
# echo of launched commands
set -x

# code execution
python -u script_exp_dataset.py dinception d ../../../data/synthetic/synth_ShapesAll_type1_nbdim_100.pickle 1000 5 8 0.8
