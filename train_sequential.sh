#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name sequential-vecnormalize
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=V100|P100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl37

cd /beegfs/desy/user/kaiserja/ares-ea-rl
python3 train_sequential.py

exit
