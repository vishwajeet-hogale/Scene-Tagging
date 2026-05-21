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
mkdir -p logs

python add_scenario_tags.py \
  --dataset_root /scratch/hogale.v/OpenLaneV2/val/10103 \
  --debug_viz_n 30 \
  --debug_viz_dir ./debug_viz_occlusion \
  --topo_bev_x_min -50.0 \
  --topo_bev_x_max 50.0 \
  --topo_bev_y_min -25.0 \
  --topo_bev_y_max 25.0
