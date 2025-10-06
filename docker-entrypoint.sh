#!/usr/bin/env bash
set -euo pipefail

# Prepare Whisper.cpp model
mkdir -p "${WHISPER_MODELS_DIR}"
MODEL_FILE="${WHISPER_MODELS_DIR}/ggml-${WHISPER_MODEL}.bin"
if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "Downloading Whisper model: ${WHISPER_MODEL}"
  /opt/whispercpp/download-ggml-model.sh "${WHISPER_MODEL}" "${WHISPER_MODELS_DIR}"
fi

# Ensure VibeVoice preloads models by default (can override externally)
export VV_PRELOAD="${VV_PRELOAD:-1}"

# Shared log for both services
LOG=/var/log/inference.log
touch "$LOG"

cd /opt/vibevoice

# Ensure exllamav3 0.0.6 is installed (from source, allow deps so flash-attn resolves)
if ! python - <<'PY' >/dev/null 2>&1; then
import sys
try:
    import exllamav3, importlib
    assert getattr(exllamav3, "__version__", "0.0.0") == "0.0.6"
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
then
  echo "[setup] Installing exllamav3==0.0.6 from source (will also install flash-attn)"
  uv pip install --system --no-build-isolation \
    "git+https://github.com/Mozer/exllamav3@v0.0.6"
fi

# Ensure SageAttention is installed (build at runtime with GPU present)
if ! python -c "import importlib; importlib.import_module('sageattention')" >/dev/null 2>&1; then
  echo "[setup] SageAttention not found. Building from source for this GPU..."
  # Detect the first GPU's compute capability via torch
  SM=$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("")
else:
    maj, min = torch.cuda.get_device_capability(0)
    print(f"{maj}.{min}")
PY
  )
  if [[ -n "$SM" ]]; then
    export TORCH_CUDA_ARCH_LIST="$SM"
    echo "[setup] Using TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
  else
    # Fallback to common 30xx/40xx archs
    export TORCH_CUDA_ARCH_LIST="8.6;8.9"
    echo "[setup] Torch could not detect GPU; falling back to $TORCH_CUDA_ARCH_LIST"
  fi
  export FORCE_CUDA=1
  # Install from the upstream repo at the chosen ref (override via env if desired)
  uv pip install --system --no-build-isolation --no-deps \
    "git+https://github.com/thu-ml/SageAttention.git@${SAGE_ATTENTION_REF:-main}" || echo "[setup] SageAttention build failed; continuing without it"
fi

# Prefetch VibeVoice assets if requested
if [[ "${VV_PRELOAD:-0}" == "1" ]]; then
  python prefetch_vibevoice.py \
    2>&1 | tee -a "$LOG"
fi

# Start Whisper.cpp server
whisper-server --host 0.0.0.0 --port 8080 --model "${MODEL_FILE}" \
  2>&1 | tee -a "$LOG" &
WHISPER_PID=$!

# Start VibeVoice Gradio wrapper (Zonos-compatible API)
python -u VIBEVOICE_ZONOS_WRAPPER.py \
  2>&1 | tee -a "$LOG" &
VV_PID=$!

# Idle watchdog with Vast.ai killswitch
echo "[watchdog] MAX_IDLE_SECONDS set to ${MAX_IDLE_SECONDS:-1800}"
(
  MAX_IDLE=${MAX_IDLE_SECONDS:-1800}
  while true; do
      last_change=$(stat -c %Y "$LOG")
      now=$(date +%s)
      idle=$(( now - last_change ))
      if (( idle > MAX_IDLE )); then
          mins=$(( idle / 60 ))
          secs=$(( idle % 60 ))
          printf "[watchdog] No log activity for %dm%02ds - stopping instance\n" "$mins" "$secs"
          if [[ -n "${CONTAINER_API_KEY:-}" && -n "${CONTAINER_ID:-}" ]]; then
            vastai set api-key "$CONTAINER_API_KEY"
            vastai stop instance "$CONTAINER_ID"
          else
            echo "[watchdog] CONTAINER_API_KEY or CONTAINER_ID not set; skipping vast.ai stop"
          fi
      fi
      sleep 60
  done
) &

# Keep container alive while services run
wait $WHISPER_PID $VV_PID
