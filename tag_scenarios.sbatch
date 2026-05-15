#!/bin/bash
#SBATCH --job-name=tag_scenarios
#SBATCH --output=logs/tag_scenarios_%j.out
#SBATCH --error=logs/tag_scenarios_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# Load necessary modules
module load cuda/12.1

# Activate your environment (adjust path if needed)
source ~/.bashrc
conda activate lanesegnet

cd /home/hogale.v/Lane_Topology/Scene_Tagging
mkdir -p logs

python add_scenario_tags.py \
  --dataset_root /scratch/hogale.v/OpenLaneV2/val \
  --topo_bev_x_min -50.0 \
  --topo_bev_x_max 50.0 \
  --topo_bev_y_min -25.0 \
  --topo_bev_y_max 25.0

python add_scenario_tags.py \
  --dataset_root /scratch/hogale.v/OpenLaneV2/train \
  --topo_bev_x_min -50.0 \
  --topo_bev_x_max 50.0 \
  --topo_bev_y_min -25.0 \
  --topo_bev_y_max 25.0

python plot_scenario_tags.py \
  --target_root /scratch/hogale.v/OpenLaneV2/val \
  --output_dir ./analysis/scenario_tag_analysis_new_val \
  --examples_per_tag 20 \
  --camera_name ring_front_center \
  --image_ext jpg
