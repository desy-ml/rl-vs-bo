#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name train-longer
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=V100|P100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl39

cd /beegfs/desy/user/kaiserja/ares-ea-rl
python3 train_longer.py

exit
