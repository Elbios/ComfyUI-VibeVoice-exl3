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

# Prefetch VibeVoice assets so Hugging Face cache is ready before server launch
python prefetch_vibevoice.py \
  2>&1 | tee -a "$LOG"

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
