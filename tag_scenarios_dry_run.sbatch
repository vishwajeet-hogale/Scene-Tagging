#!/bin/bash
#SBATCH --job-name=tag_scenarios_dry_run
#SBATCH --output=logs/tag_scenarios_dry_run_%j.out
#SBATCH --error=logs/tag_scenarios_dry_run_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Load necessary modules
module load cuda/12.1

# Activate environment
source ~/.bashrc
conda activate lanesegnet

cd /home/hogale.v/Lane_Topology/Scene_Tagging

python add_scenario_tags.py \
  --dataset_root /scratch/hogale.v/OpenLaneV2/val/10043 \
  --dry_run
