#!/bin/sh
#SBATCH --partition=maxwell
#SBATCH --job-name gamma-sweep
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl

cd /beegfs/desy/user/kaiserja/ares-ea-rl
wandb agent msk-ipc/ares-ea-rl-a-new-hope/zwhxczsw

exit
