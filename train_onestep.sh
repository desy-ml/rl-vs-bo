#!/bin/sh
# #SBATCH --partition=maxgpu
#SBATCH --partition=maxwell
#SBATCH --job-name onestep-new-gym
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
# #SBATCH --constraint=V100|P100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl37

cd /beegfs/desy/user/kaiserja/ares-ea-rl
python train_onestep.py

exit
