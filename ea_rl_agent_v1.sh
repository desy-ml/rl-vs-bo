#!/usr/bin/bash

# Script for running the Jupyter notebook from the jddd Cockpit

. /home/aresoper/.profile
conda activate /home/kaiserja/.conda/envs/rl39ng
cd /home/aresoper/user/kaiserja/ares-ea-v2
jupyter lab ea_rl_agent_v1.ipynb
