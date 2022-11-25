#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name ares-ea-v2
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=P100|V100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl39
cd /beegfs/desy/user/kaiserja/ares-ea-v2

# python ea_train.py
wandb agent --count 1 msk-ipc/ares-ea-v2/gun6p5gd

exit
