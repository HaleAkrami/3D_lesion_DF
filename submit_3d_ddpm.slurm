#!/bin/bash
#SBATCH --account=ajoshi_27
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
eval "$(conda shell.bash hook)"
conda activate working_monai
python3 3d_ddpm.py
