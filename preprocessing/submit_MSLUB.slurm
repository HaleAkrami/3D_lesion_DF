#!/bin/bash
#SBATCH --account=ajoshi_27
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
eval "$(conda shell.bash hook)"
conda activate plt-uad
bash prepare_MSLUB.sh /scratch1/akrami/Data_pre/MSLUB /scratch1/akrami/Data_T1


