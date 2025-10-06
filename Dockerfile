# ------------------------------------------------------------------
# VibeVoice TTS + Whisper.cpp Server Image (Higgs-style)
# ------------------------------------------------------------------

# Use the same starter image as the Higgs dockerfile example.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel AS runtime-base

LABEL description="VibeVoice TTS + Whisper.cpp CUDA server"

# Whisper.cpp version
ARG WCPP_VER=v1.7.6
WORKDIR /opt

# OS-level deps (match Higgs image) for whisper.cpp build and audio runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        ninja-build \
        ffmpeg \
        portaudio19-dev \
        libasound2 \
        espeak-ng \
        wget \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Build whisper.cpp with CUDA
RUN git clone --depth 1 --branch ${WCPP_VER} https://github.com/ggml-org/whisper.cpp.git
WORKDIR /opt/whisper.cpp
RUN cmake -B build -DGGML_CUDA=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="50;60;61;70;75;80;86;89;90" \
 && cmake --build build -j $(nproc) --config Release
WORKDIR /opt

# Base utilities (same as Higgs example)
RUN python -m pip install --no-cache-dir --upgrade pip uv vastai huggingface_hub "python-dateutil>=2.8.2"

# ------------------------------------------------------------------
# App layer: copy repo and install Python deps
# ------------------------------------------------------------------
FROM runtime-base AS app

WORKDIR /opt/vibevoice

# Copy repository into the container
COPY . /opt/vibevoice

# Install Python dependencies
# - Use uv for speed and system site-packages inside container
# - Pin user-tested versions for attention backends and exllamav3
# - Keep gradio stack consistent with Higgs example
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN uv pip install --system -r requirements.txt && \
    uv pip install --system \
        gradio==5.38.2 \
        gradio-client==1.11.0 \
        fastapi==0.115.14 \
        starlette==0.45.3 && \
    uv pip install --system transformers==4.54.1 diffusers formatron && \
    uv cache clean

# Prepare whispercpp runtime files
RUN mkdir -p /opt/whispercpp/models && \
    mv /opt/whisper.cpp/build/bin/* /opt/whispercpp/ && \
    mv /opt/whisper.cpp/build/src/libwhisper.so* /usr/local/lib/ && \
    mv /opt/whisper.cpp/build/ggml/src/libggml* /usr/local/lib/ && \
    mv /opt/whisper.cpp/build/ggml/src/ggml-cuda/libggml-cuda.so /usr/local/lib/ && \
    mv /opt/whisper.cpp/models/download-ggml-model.sh /opt/whispercpp/ && \
    rm -rf /opt/whisper.cpp && \
    ldconfig && \
    ln -s /opt/whispercpp/whisper-server /usr/local/bin/whisper-server && \
    ln -s /opt/whispercpp/whisper-cli    /usr/local/bin/whisper

# Entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Whisper defaults (same as Higgs style)
ENV WHISPER_MODEL=medium-en \
    WHISPER_MODELS_DIR=/opt/whispercpp/models

# Expose all VV tunables so they’re visible/overridable at run time
# Defaults mirror VIBEVOICE_ZONOS_WRAPPER.py
ENV VV_MODEL=VibeVoice-7B-no-llm-bf16 \
    VV_EXLLAMA_MODEL=vibevoice-7b-exl3-8bit \
    VV_ATTENTION=sage \
    VV_QUANT=bf16 \
    VV_DIFFUSION_STEPS=5 \
    VV_NEG_STEPS_CACHE=2 \
    VV_INCREASE_CFG=1 \
    VV_TEMPERATURE=0.95 \
    VV_FREE_MEMORY_AFTER_JOB=0 \
    VV_USE_SAMPLING=0 \
    VV_PRELOAD=0 \
    VV_FORCE_SEED=1 \
    VV_FIXED_SEED=42

# Network ports (Gradio + Whisper server)
EXPOSE 7860 8080

ENTRYPOINT ["docker-entrypoint.sh"]
