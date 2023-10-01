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
bash prepare_biobank.sh /scratch1/akrami/Latest_Data/Data_pre/BioBank_val /scratch1/akrami/Latest_Data/Data_T1


