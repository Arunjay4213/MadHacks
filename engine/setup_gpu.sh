#!/usr/bin/env bash
# Reproduce the GPU engine environment on a fresh Ubuntu 24.04 machine with an NVIDIA driver.
# Layout: third-party code in ~/third_party, weights in ~/weights, venv in ~/engine-venv.
# No tokens are needed: every repo used here is public and ungated.
# Usage: bash engine/setup_gpu.sh            (downloads run in the foreground; run it inside tmux)
set -euo pipefail

sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg ripgrep fd-find tmux git jq
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv tool install "huggingface_hub[cli,hf_transfer]" || true
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p ~/third_party ~/weights/taehv

# Official Causal Forcing code (Apache-2.0). Pinned to the commit that was benchmarked.
if [ ! -d ~/third_party/Causal-Forcing ]; then
  git clone https://github.com/thu-ml/Causal-Forcing.git ~/third_party/Causal-Forcing
fi
git -C ~/third_party/Causal-Forcing checkout -q eb8705b2f00b7c361193b25aa3aae9e7edc94667
# The official code opens weights through these relative paths.
ln -sfn ~/weights ~/third_party/Causal-Forcing/wan_models
ln -sfn ~/weights/Causal-Forcing ~/third_party/Causal-Forcing/checkpoints

# Weights (all Apache-2.0 on Hugging Face, ungated). Run in parallel.
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ~/weights/Wan2.1-T2V-1.3B &
hf download zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt chunkwise/causal_forcing.pt README.md \
  --local-dir ~/weights/Causal-Forcing &
# Tiny VAE decoder used by the official demo (MIT, github.com/madebyollin/taehv).
curl -sL -o ~/weights/taehv/taew2_1.pth https://github.com/madebyollin/taehv/raw/main/taew2_1.pth &
wait

# Python env. Python 3.10 because the official requirements pin numpy 1.24.4.
uv venv --python 3.10 ~/engine-venv
# shellcheck disable=SC1090
source ~/engine-venv/bin/activate
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -r "$(dirname "$0")/requirements-gpu.txt"
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__, torch.cuda.get_device_name(0))"
echo "setup done"
