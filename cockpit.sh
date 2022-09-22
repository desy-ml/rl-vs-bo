#!/usr/bin/bash

# Script for running the RL Agent EA app from the ARES cockpit

. /home/aresoper/.profile
conda activate /home/kaiserja/.conda/envs/rl39ng
cd /home/aresoper/user/kaiserja/ares-ea-v2
python app.py
