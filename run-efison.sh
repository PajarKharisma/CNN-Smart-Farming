#!/bin/bash
#SBATCH --account=cbt
#SBATCH --partition=zentwo
#SBATCH --job-name=job
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=log/result/result-%j.out
#SBATCH --error=log/result/result-%j.err

module load anaconda3
eval “$(conda shell.bash hook)”
conda activate $WORK/.venv

#%Module
module load cuda/10.2-cuDNN7.6.5
module load tensorrt/6-cuda10.2

python3 src/main/main.py
