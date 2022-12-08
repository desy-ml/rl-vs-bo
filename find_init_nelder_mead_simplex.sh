#!/bin/sh
#SBATCH --partition=maxwell
#SBATCH --job-name ares-ea-v2
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl39
cd /beegfs/desy/user/kaiserja/ares-ea-v2

wandb agent --count 1 msk-ipc/ares-ea-nelder-mead-init-simplex/i36mm3is

exit
