#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name aresearlsweep
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=V100|P100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate /beegfs/desy/user/kaiserja/anaconda3/envs/rl37

cd /beegfs/desy/user/$USER/ares-ea-rl
wandb agent --count 1 msk-ipc/ares-ea-rl-a-new-hope/yj9w9jx1

exit
