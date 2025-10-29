# ----------------------------------------------------------
#  Base: CUDA 11.8 + cuDNN8 + Ubuntu 22.04
# ----------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ---- 基本設定 ----
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ENV HF_HOME=/app/hf_cache
ENV VLLM_LOGGING_LEVEL=INFO
ENV VLLM_PLATFORM=cpu  

WORKDIR /app

# ---- 安裝基本套件 ----
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip git wget curl vim \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && python -m pip install --upgrade pip setuptools wheel \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- 安裝 PyTorch / Transformers / Flash-Attn / vLLM ----
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    transformers==4.46.3 \
    tokenizers==0.20.3 \
    einops addict easydict \
    psutil

RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu118

# ---- 安裝 vLLM (nightly) ----
RUN pip install -U "vllm>=0.11.0.dev0" --pre --extra-index-url https://wheels.vllm.ai/nightly

# ---- 安裝 OCR / FastAPI 依賴 ----
RUN pip install --no-cache-dir pillow==10.4.0 \
    matplotlib==3.9.2 \
    fastapi uvicorn pydantic

# ---- 建立模型快取資料夾 ----
RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

# ---- 自動下載 DeepSeek-OCR 模型（強制下載，每次都覆蓋） ----
RUN python - <<'EOF'
import os
from huggingface_hub import snapshot_download
import shutil

model_name = "deepseek-ai/deepseek-ocr"
model_dir = "/app/models/deepseek-ocr"

# 如果資料夾已存在，先刪掉
if os.path.exists(model_dir):
    print(f"Removing existing model at {model_dir} ...")
    shutil.rmtree(model_dir)

os.makedirs(model_dir, exist_ok=True)
print(f"Downloading {model_name} to {model_dir} ...")
snapshot_download(repo_id=model_name, local_dir=model_dir)
EOF

# ---- 容器啟動命令 ----
CMD ["uvicorn", "serve_ocr:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "critical", "--reload"]