#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:10:00
#SBATCH --job-name=install-mamba-ssm
#SBATCH --output=logs/install-mamba-ssm-%j.log
#SBATCH --error=logs/install-mamba-ssm-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --environment=alps3
#SBATCH --no-requeue

mkdir -p logs

echo "[$(date)] Installing mamba-ssm..."
pip install mamba-ssm --user
echo "[$(date)] Verifying install..."
python -c "import mamba_ssm; print('mamba-ssm version:', mamba_ssm.__version__)"
echo "[$(date)] Done."
