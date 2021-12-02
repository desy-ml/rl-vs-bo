#!/bin/bash

echo "Sourcing .bashrc"
source ~/.bashrc

echo "Activating conda environment"
conda activate /home/kaiserja/.conda/envs/rl

echo "Setting EARLMCP environment variable to write to PyDOOCS"
export EARLMCP=pydoocs

echo "Runnig Python script"
python evaluate_machine_agents.py
