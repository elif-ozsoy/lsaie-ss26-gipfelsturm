#!/bin/bash
#
# Test the graceful SIGTERM exit mechanism end-to-end.
#
# Launches a short training run on 1 node using the 125m model.
# After a configurable delay the job sends SIGTERM to itself, simulating
# what SLURM does at walltime. No manual scancel timing required.
#
# Usage: ./test-graceful-exit.sh [sigterm_delay_seconds]
#
# sigterm_delay_seconds: how long to wait before firing SIGTERM (default 120).
#   Should be long enough for initialization + a few training steps.
#   On Clariden, 125m initialization takes ~30–60 s, so 120 s is safe.
#
# What to check in the log:
#   PASS: "exiting program after receiving SIGTERM" appears, job ends cleanly
#   FAIL: log cuts off mid-output with no SIGTERM message (hard kill)

set -euo pipefail

source "$(dirname "$0")/config.sh"

SIGTERM_DELAY=${1:-120}

mkdir -p logs
SCRIPT="logs/test-graceful-exit.sbatch"

cat > "$SCRIPT" << SBATCH_HEAD
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=00:10:00
#SBATCH --job-name=test-graceful-exit
#SBATCH --output=logs/test-graceful-exit-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
echo "START TIME: \$(date)"
echo "SIGTERM will fire in ${SIGTERM_DELAY}s after this job starts."

WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << 'SETUP'
mkdir -p logs $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRAINING_CMD="torchrun \
    --nproc-per-node $SLURM_GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    $MEGATRON_LM_DIR/pretrain_gpt.py \
    --transformer-impl transformer_engine \
    --use-precision-aware-optimizer \
    --main-grads-dtype bf16 \
    --num-layers 12 \
    --hidden-size 768 \
    --ffn-hidden-size 2048 \
    --num-attention-heads 12 \
    --group-query-attention \
    --num-query-groups 4 \
    --max-position-embeddings 4096 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --seq-length 4096 \
    --micro-batch-size 4 \
    --global-batch-size 64 \
    --train-iters 200 \
    --log-interval 1 \
    --eval-interval 200 \
    --eval-iters 0 \
    --disable-bias-linear \
    --optimizer adam \
    --dataloader-type single \
    --no-check-for-nan-in-loss-and-grad \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --lr 3e-4 \
    --lr-decay-style constant \
    --lr-warmup-iters 10 \
    --seed 42 \
    --init-method-std 0.02 \
    --bf16 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --exit-signal-handler \
    --log-throughput \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file $WORKDIR/data/gpt2-vocab.json \
    --merge-file $WORKDIR/data/gpt2-merges.txt \
    --data-path $DATA_PREFIX \
    --data-cache-path $DATASET_CACHE_DIR \
    --split 99,1,0 \
    --num-workers 1"
SETUP

cat >> "$SCRIPT" << TIMER
# Background timer: sends SIGTERM to the job after the configured delay.
# This simulates SLURM's walltime signal without requiring manual scancel.
# Killed automatically if training finishes normally before the timer fires.
(
    sleep ${SIGTERM_DELAY}
    echo ""
    echo "[\$(date)] >>> TEST: firing SIGTERM after ${SIGTERM_DELAY}s delay <<<"
    scancel --signal=TERM \$SLURM_JOB_ID
) &
TIMER_PID=\$!
trap "kill \$TIMER_PID 2>/dev/null; wait \$TIMER_PID 2>/dev/null || true" EXIT

echo "CMD: \$TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task \$SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 \$TRAINING_CMD"
TIMER

cat >> "$SCRIPT" << 'FOOTER'

# If srun returns here, check whether it was a graceful exit or normal completion.
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo ">>> srun exited cleanly (code 0) <<<"
    echo "Check the log above for 'exiting program after receiving SIGTERM'."
    echo "If that line is present: PASS. If training ran all 200 steps: timer didn't fire in time."
else
    echo ""
    echo ">>> srun exited with code $EXIT_CODE <<<"
    echo "This may indicate a hard kill rather than graceful exit."
fi

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"
echo "Generated: $SCRIPT"
echo "SIGTERM will fire ${SIGTERM_DELAY}s after the job starts."
echo ""
sbatch "$SCRIPT"
