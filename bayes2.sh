#!/bin/sh
#SBATCH --partition=maxwell
#SBATCH --job-name aresea-random-search
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl39

cd /beegfs/desy/user/kaiserja/ares-ea-rl
python3 bayes2.py

exit
