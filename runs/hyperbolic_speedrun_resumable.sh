#!/usr/bin/env bash
set -euo pipefail

# Resumable FineWeb + base model training for Hyperbolic GPU instances.
# Defaults are tuned for 8xH100, but can be overridden via env vars below.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

MODEL_TAG="${MODEL_TAG:-d26-hyperbolic}"
WANDB_RUN="${WANDB_RUN:-dummy}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

DEPTH="${DEPTH:-26}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8.5}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
SAVE_EVERY="${SAVE_EVERY:-250}"

FINEWEB_BOOTSTRAP_SHARDS="${FINEWEB_BOOTSTRAP_SHARDS:-8}"
FINEWEB_TOTAL_SHARDS="${FINEWEB_TOTAL_SHARDS:-370}"

RUN_BASE_EVAL="${RUN_BASE_EVAL:-1}"
RUN_SFT="${RUN_SFT:-1}"
RUN_CHAT_EVAL="${RUN_CHAT_EVAL:-1}"

HF_REPO_ID="${HF_REPO_ID:-}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"  # model or dataset
HF_PRIVATE="${HF_PRIVATE:-true}"
HF_SYNC_INTERVAL="${HF_SYNC_INTERVAL:-1200}"
HF_DO_RESTORE="${HF_DO_RESTORE:-1}"

mkdir -p "$NANOCHAT_BASE_DIR"

install_uv_if_missing() {
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

setup_python_env() {
  install_uv_if_missing
  [ -d ".venv" ] || uv venv
  uv sync --extra gpu
  source .venv/bin/activate
  uv pip install --quiet huggingface_hub
}

latest_checkpoint_step() {
  local checkpoint_dir="$1"
  if ls "$checkpoint_dir"/model_*.pt >/dev/null 2>&1; then
    ls "$checkpoint_dir"/model_*.pt \
      | sed -E 's/.*model_([0-9]+)\.pt/\1/' \
      | sort -n \
      | tail -n 1
  fi
}

tokenizer_exists() {
  [ -d "$NANOCHAT_BASE_DIR/tokenizer" ] && [ -n "$(ls -A "$NANOCHAT_BASE_DIR/tokenizer" 2>/dev/null)" ]
}

SYNC_PID=""
do_final_sync() {
  if [[ -n "$HF_REPO_ID" ]]; then
    python ops/hyperbolic/hf_sync_loop.py \
      --repo-id "$HF_REPO_ID" \
      --repo-type "$HF_REPO_TYPE" \
      --private "$HF_PRIVATE" \
      --local-dir "$NANOCHAT_BASE_DIR" \
      --once || true
  fi
}

cleanup() {
  local exit_code=$?
  if [[ -n "$SYNC_PID" ]]; then
    kill "$SYNC_PID" >/dev/null 2>&1 || true
    wait "$SYNC_PID" >/dev/null 2>&1 || true
  fi
  do_final_sync
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

setup_python_env

# Optional auth:
# HF_TOKEN can be exported before launch. If omitted, existing `hf auth login` state is used.
if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
fi

# Optional restore from HF backups.
if [[ -n "$HF_REPO_ID" && "$HF_DO_RESTORE" == "1" ]]; then
  python ops/hyperbolic/hf_restore.py \
    --repo-id "$HF_REPO_ID" \
    --repo-type "$HF_REPO_TYPE" \
    --local-dir "$NANOCHAT_BASE_DIR" \
    --skip-if-local
fi

# Optional background checkpoint sync to HF while training runs.
if [[ -n "$HF_REPO_ID" ]]; then
  mkdir -p "$NANOCHAT_BASE_DIR/logs"
  python ops/hyperbolic/hf_sync_loop.py \
    --repo-id "$HF_REPO_ID" \
    --repo-type "$HF_REPO_TYPE" \
    --private "$HF_PRIVATE" \
    --local-dir "$NANOCHAT_BASE_DIR" \
    --interval-seconds "$HF_SYNC_INTERVAL" \
    > "$NANOCHAT_BASE_DIR/logs/hf_sync.log" 2>&1 &
  SYNC_PID="$!"
  echo "HF sync loop started: pid=$SYNC_PID, log=$NANOCHAT_BASE_DIR/logs/hf_sync.log"
fi

CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
mkdir -p "$CHECKPOINT_DIR"
RESUME_STEP="$(latest_checkpoint_step "$CHECKPOINT_DIR" || true)"

if [[ -z "$RESUME_STEP" ]]; then
  echo "No existing checkpoint found for model tag '$MODEL_TAG'. Starting fresh run."
  python -m nanochat.report reset
else
  echo "Found checkpoint at step $RESUME_STEP for model tag '$MODEL_TAG'. Will resume."
fi

# FineWeb download + tokenizer.
if ! tokenizer_exists; then
  python -m nanochat.dataset -n "$FINEWEB_BOOTSTRAP_SHARDS"
  python -m scripts.tok_train
  python -m scripts.tok_eval
else
  echo "Tokenizer already exists at $NANOCHAT_BASE_DIR/tokenizer, skipping tokenizer training."
fi

# Keep downloading shards while continuing pipeline.
python -m nanochat.dataset -n "$FINEWEB_TOTAL_SHARDS" &
DATASET_PID="$!"

echo "Waiting for FineWeb shard download to finish..."
wait "$DATASET_PID"

RESUME_ARGS=()
if [[ -n "$RESUME_STEP" ]]; then
  RESUME_ARGS+=(--resume-from-step "$RESUME_STEP")
fi

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
  --depth="$DEPTH" \
  --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --fp8 \
  --save-every="$SAVE_EVERY" \
  --model-tag="$MODEL_TAG" \
  --run="$WANDB_RUN" \
  "${RESUME_ARGS[@]}"

if [[ "$RUN_BASE_EVAL" == "1" ]]; then
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$MODEL_TAG"
fi

if [[ "$RUN_SFT" == "1" ]]; then
  curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN"
  if [[ "$RUN_CHAT_EVAL" == "1" ]]; then
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
      -i sft -g "$MODEL_TAG"
  fi
fi

python -m nanochat.report generate
echo "Run complete."
