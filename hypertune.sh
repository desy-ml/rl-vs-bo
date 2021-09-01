#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name aresearlsweep
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=V100|P100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl37

cd /beegfs/desy/user/kaiserja/ares-ea-rl
wandb agent --count 1 msk-ipc/ares-ea-rl-a-new-hope/2zh2ml8p

exit
