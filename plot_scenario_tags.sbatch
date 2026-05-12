#!/bin/bash
#SBATCH --job-name=plot_scenario_tags
#SBATCH --output=logs/plot_scenario_tags_%j.out
#SBATCH --error=logs/plot_scenario_tags_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00

# Activate environment
source ~/.bashrc
conda activate lanesegnet

cd /home/hogale.v/Lane_Topology/Scene_Tagging

python plot_scenario_tags.py \
  --target_root /scratch/hogale.v/OpenLaneV2/val \
  --output_dir ./analysis/scenario_tag_analysis_new_val
