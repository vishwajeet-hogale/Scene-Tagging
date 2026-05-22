#!/bin/bash
#SBATCH --job-name=analyze_taxonomy
#SBATCH --output=logs/analyze_taxonomy_%j.out
#SBATCH --error=logs/analyze_taxonomy_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00

source ~/.bashrc
conda activate lanesegnet

cd /home/hogale.v/Lane_Topology/Scene_Tagging

mkdir -p logs analysis/taxonomy_train analysis/taxonomy_val

echo "[$(date)] Running train split..."
python analyze_taxonomy.py \
  --dataset_root /scratch/hogale.v/OpenLaneV2/train \
  --output_dir ./analysis/taxonomy_train \
  # --skip_plots

echo "[$(date)] Running val split..."
python analyze_taxonomy.py \
  --dataset_root /scratch/hogale.v/OpenLaneV2/val \
  --output_dir ./analysis/taxonomy_val \
  # --skip_plots

python analyze_taxonomy.py \
  --dataset_root /scratch/hogale.v/OpenLaneV2/train \
  --output_dir ./analysis/taxonomy_train \

echo "[$(date)] Done."
