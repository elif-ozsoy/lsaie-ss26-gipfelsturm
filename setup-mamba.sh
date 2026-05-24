#!/bin/bash
#
# One-time setup: install causal-conv1d and mamba-ssm inside the alps3 container
# on a compute node (ARM64/GH200), writing packages to ~/.local.
#
# Run this once before using launch-mamba.sh. Check the job log in logs/ to confirm
# the installation succeeded.
#
# Usage: ./setup-mamba.sh

set -euo pipefail

source "$(dirname "$0")/config.sh"

mkdir -p logs
SCRIPT="logs/setup-mamba-deps.sbatch"

cat > "$SCRIPT" << SBATCH
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=00:30:00
#SBATCH --job-name=setup-mamba-deps
#SBATCH --output=logs/setup-mamba-deps-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --no-requeue

echo "START TIME: \$(date)"
echo "Installing causal-conv1d and mamba-ssm inside alps3 container (ARM64)..."

# Run pip inside the container where CUDA and Python are available.
# --user installs to ~/.local, which is accessible via the /users mount.
# --force-reinstall is required: causal-conv1d may already exist in ~/.local
# compiled for x86 or a different CUDA version (e.g. installed on a login node).
# Without it, pip skips reinstallation and the wrong .so is still loaded on GH200.
srun -n1 --environment=alps3 python3 -m pip install --user --force-reinstall \\
    "causal-conv1d>=1.4.0" \\
    "mamba-ssm>=2.2.2"

echo "Verifying installation..."
srun -n1 --environment=alps3 python3 -c "
import causal_conv1d; print('causal_conv1d:', causal_conv1d.__version__)
import mamba_ssm;     print('mamba_ssm:    ', mamba_ssm.__version__)
from mamba_ssm.ops.triton.ssd_combined import causal_conv1d_fwd_function
assert causal_conv1d_fwd_function is not None, 'CUDA extension not loaded!'
print('causal_conv1d_fwd_function: OK')
"

echo "END TIME: \$(date)"
SBATCH

chmod +x "$SCRIPT"
echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
