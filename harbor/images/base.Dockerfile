# Shared base for the agent, eval-server and verifier images: Python env, base model, offline training data and the
# reference repo. No eval code, targets or eval data: those live only in the eval-server and verifier images.
# Built by harbor/build_images.sh from a staged context.
FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# build-essential: Triton compiles a small C launcher at runtime (flash-linear-attention kernels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev build-essential git curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.3 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/bin/python3.10 UV_LINK_MODE=copy

WORKDIR /opt/build
COPY pyproject.toml uv.lock ./
# flash-attn and causal-conv1d are copied prebuilt from the host venv (same Python/torch/CUDA) to skip a long compile
RUN uv sync --frozen --extra linear-attn --no-install-project \
        --no-install-package flash-attn --no-install-package causal-conv1d \
    && rm -rf /root/.cache/uv
COPY prebuilt/ /opt/venv/lib/python3.10/site-packages/
# COPY keeps host permissions (umask 007 there), and the verifier trains as an unprivileged user
RUN chmod -R a+rX /opt/venv

ENV PATH=/opt/venv/bin:$PATH HF_HOME=/opt/hf
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B')" \
    && python -c "from datasets import load_dataset; \
load_dataset('facebook/md_gender_bias', name='funpedia', trust_remote_code=True); \
load_dataset('stanfordnlp/snli'); load_dataset('stanfordnlp/sst2')" \
    && chmod -R a+rwX /opt/hf

COPY ao_data/ /opt/ao_data/
COPY ao_ref/ /opt/ao_ref/
RUN chmod -R a+rX /opt/ao_data /opt/ao_ref

ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
