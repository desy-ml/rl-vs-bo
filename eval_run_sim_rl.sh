#!/bin/sh
#SBATCH --partition=maxwell
#SBATCH --job-name rl-second5-37-4
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl39

cd /beegfs/desy/user/kaiserja/ares-ea-rl
python3 eval_run_sim_rl.py

exit
