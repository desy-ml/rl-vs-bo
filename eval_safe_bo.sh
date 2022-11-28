#!/bin/sh
#SBATCH --partition=maxwell
#SBATCH --job-name eval-safe-bo
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
##SBATCH --constraint=P100|V100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl39
cd /beegfs/desy/user/kaiserja/ares-ea-v2

module load matlab

matlab -nosplash -nodesktop -nojvm -r start_matlab &
python eval_safe_bo_sim.py

exit
