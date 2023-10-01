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
bash prepare_Brats_v2.sh /scratch1/akrami/Latest_Data/Data_pre/Brats21 /scratch1/akrami/Latest_Data/T1_T2_v2


