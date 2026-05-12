#!/bin/bash
#SBATCH --job-name=tag_scenarios
#SBATCH --output=logs/tag_scenarios_%j.out
#SBATCH --error=logs/tag_scenarios_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=8:00:00

# Load necessary modules
module load cuda/12.1

# Activate your environment (adjust path if needed)
source ~/.bashrc
conda activate lanesegnet

# cd /home/hogale.v/Lane_Topology/Scene_Tagging

# python add_scenario_tags.py \
#   --dataset_root /scratch/hogale.v/OpenLaneV2/val

# python add_scenario_tags.py \
#   --dataset_root /scratch/hogale.v/OpenLaneV2/train

cd /home/hogale.v/Lane_Topology/OLV2_Scenario_Tag_Gen

python ./tools/data_process.py

python tools/eval_per_scenario.py projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py results/results.pkl --out-dir results/scenario_eval_new_curv_lane_greedy --meta-pkl /scratch/hogale.v/OpenLaneV2/data_dict_subset_A_val_lanesegnet.pkl

python tools/eval_per_scenario_hungarian.py projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py results/results.pkl --out-dir results/scenario_eval_hungarian_all --meta-pkl /scratch/hogale.v/OpenLaneV2/data_dict_subset_A_val_lanesegnet.pkl
