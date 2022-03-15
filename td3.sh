#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name train-td3
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=V100|P100
#SBATCH --mail-type ALL

source ~/.bashrc
conda activate rl39

cd /beegfs/desy/user/kaiserja/ares-ea-rl
python td3.py

exit
