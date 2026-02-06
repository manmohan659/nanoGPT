# Hyperbolic + FineWeb + Resumable Training

This setup gives you:
- FineWeb pretraining data (`karpathy/fineweb-edu-100b-shuffle`)
- periodic local checkpoints (`--save-every`)
- automatic resume from latest checkpoint
- optional backup/restore of checkpoints to Hugging Face

## 1) Hyperbolic instance setup

1. Create a GPU instance.
2. Attach persistent storage (Hyperbolic Network Volume) and mount it.
3. Use the mount path as `NANOCHAT_BASE_DIR`, for example `/mnt/hv/nanochat`.

Why: if the instance dies, your training state remains on persistent storage.

## 2) Clone repo on the instance

You do not need to fork to train.

```bash
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

Fork only if you want to maintain your own code changes in GitHub.

## 3) Connect Hugging Face + GitHub

Hugging Face (for checkpoints/artifacts):

```bash
export HF_TOKEN=hf_xxx
hf auth login --token "$HF_TOKEN" --add-to-git-credential
```

GitHub (for code/scripts only, not checkpoints):

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
# If you use SSH:
# ssh-keygen -t ed25519 -C "you@example.com"
# then add key to GitHub and use git@github.com:...
```

## 4) Launch resumable training

```bash
export NANOCHAT_BASE_DIR=/mnt/hv/nanochat
export HF_REPO_ID=<hf-username>/nanochat-checkpoints
export HF_REPO_TYPE=model
export HF_PRIVATE=true
export WANDB_RUN=hyperbolic-d26

bash runs/hyperbolic_speedrun_resumable.sh
```

## 5) Resume after crash/restart

Run the same command again:

```bash
bash runs/hyperbolic_speedrun_resumable.sh
```

The script:
- restores from HF (if enabled and local checkpoints are missing),
- detects latest local checkpoint step for `MODEL_TAG`,
- resumes with `--resume-from-step`.

## Useful env vars

- `MODEL_TAG` default: `d26-hyperbolic`
- `SAVE_EVERY` default: `250`
- `NPROC_PER_NODE` default: `8`
- `RUN_SFT` default: `1`
- `HF_SYNC_INTERVAL` default: `1200` (20 min)

Example:

```bash
MODEL_TAG=d24-prod SAVE_EVERY=100 RUN_SFT=0 bash runs/hyperbolic_speedrun_resumable.sh
```

## Monitoring

```bash
# latest checkpoints
ls -lh "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" | tail

# HF sync loop logs
tail -f "$NANOCHAT_BASE_DIR/logs/hf_sync.log"
```

## Important notes

- Keep GPU count consistent when resuming base training. Optimizer state is saved per rank.
- GitHub is not suitable for multi-GB checkpoint files; use HF Hub for weights.
- If you terminate an instance without persistent storage, local data is lost.
