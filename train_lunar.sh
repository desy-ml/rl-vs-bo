#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name lunartrain
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=V100|P100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh
module load anaconda3
module load cuda

cd ~/ares-ea-rl
python3 train_lunar.py

exit
