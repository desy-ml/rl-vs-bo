#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name ares-ea-v2
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=A100|P100|V100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate rl39
cd /beegfs/desy/user/kaiserja/ares-ea-v2

python train.py --action_type direct --filter_action 0 1 3 --filter_observation beam magnets --incoming random --magnet_init random --misalignments random --reward_method feedback --quad_action oneway --sb3_device cpu --target_beam_mode random --vec_env dummy --w_mu_x 0.0 --w_mu_y 0.0 --w_time 0.0

exit
