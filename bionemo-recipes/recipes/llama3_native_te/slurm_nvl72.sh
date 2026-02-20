#!/bin/bash
#SBATCH --nodes=9                         # number of nodes
#SBATCH --segment=9
#SBATCH --ntasks-per-node=1                 # one task per node; torchrun handles per-GPU processes
#SBATCH --time=00:20:00                     # wall time
#SBATCH --mem=0                             # all mem avail
#SBATCH --account=healthcareeng_bionemo     # account
#SBATCH --partition=gb300          # partition
#SBATCH --mail-type=FAIL                    # only send email on failure
#SBATCH --exclusive                         # exclusive node access
#SBATCH --output=job_output/slurm_%x.%j.out
#SBATCH --job-name=healthcareeng_bionemo-recipes.llama3-cp-benchmark

set -x -e
ulimit -c 0

# Usage:
#   sbatch slurm_nvl72.sh        # defaults to cp=6
#   sbatch slurm_nvl72.sh 9      # cp_size=9, seq_len=73728 (9*8192)

# Accept cp size as first positional argument (default: 6)
CP_SIZE=${1:-6}
MAX_SEQ_LENGTH=$((CP_SIZE * 8192))
SEQ_LENGTH_K=$((MAX_SEQ_LENGTH / 1000))K

export CMD="TRITON_CACHE_DIR=/tmp/triton_cache \
    HF_TOKEN=hf_aK... \
    NVTE_BATCH_MHA_P2P_COMM=1 \
    torchrun \
    --rdzv_id \$SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint \$MASTER_ADDR:\$MASTER_PORT \
    --nproc-per-node 4 \
    --nnodes \$SLURM_NNODES \
    --node-rank \$SLURM_NODEID \
    train_fsdp2_cp.py \
    --config-name L2_cp_benchmark \
    wandb.name=llama3-cp-benchmark-lyris-32gpu-cp${CP_SIZE}-70B-${SEQ_LENGTH_K} \
    wandb.project=bionemo-recipes-pstjohn \
    wandb.mode=online \
    dataset.max_seq_length=${MAX_SEQ_LENGTH} \
    cp_size=${CP_SIZE} \
    config_name_or_path=meta-llama/Llama-3.1-70B
"

srun \
  --container-image=/lustre/fsw/healthcareeng_bionemo/pstjohn/enroot/nvidian+cvai_bnmo_trng+bionemo+llama3_cp_arm_0211.sqsh \
  --container-mounts=$HOME/.netrc:/root/.netrc,/lustre/fsw/healthcareeng_bionemo/pstjohn/cache:/root/.cache \
  bash -c "$CMD"
