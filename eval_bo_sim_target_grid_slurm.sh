#!/bin/sh
#SBATCH --partition=allcpu
#SBATCH --job-name bo-target-grid
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl39
cd /beegfs/desy/user/kaiserja/ares-ea-v2

python eval_bo_sim_target_grid.py

exit
