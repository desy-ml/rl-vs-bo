#!/bin/sh
#SBATCH --partition=maxwell
#SBATCH --job-name stationary-conditional-objective
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl

cd /beegfs/desy/user/kaiserja/ares-ea-rl
python3 train.py

exit
