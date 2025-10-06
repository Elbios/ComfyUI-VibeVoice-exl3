"""VibeVoice TTS Gradio wrapper compatible with the legacy Zonos/Higgs client API."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import librosa
import numpy as np
import soundfile as sf

from nodes.base_vibevoice import BaseVibeVoiceNode, SAGE_AVAILABLE

logger = logging.getLogger("VibeVoiceWrapper")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[VibeVoiceWrapper] %(message)s'))
    logger.addHandler(handler)


class VibeVoiceService:
    """Thin service that reuses the ComfyUI VibeVoice node outside of ComfyUI."""

    def __init__(self) -> None:
        self._node = BaseVibeVoiceNode()
        self._model_loaded = False
        self._active_config: Optional[Tuple[str, str, str, str]] = None

        # Default configuration mirrors the tested ComfyUI setup.
        self._model_handle = os.getenv("VV_MODEL", "VibeVoice-7B-no-llm-bf16")
        self._exllama_handle = os.getenv("VV_EXLLAMA_MODEL", "vibevoice-7b-exl3-8bit")
        default_attn = "sage" if SAGE_AVAILABLE else "auto"
        self._attention = os.getenv("VV_ATTENTION", default_attn)
        if self._attention == "sage" and not SAGE_AVAILABLE:
            logger.warning("SageAttention not available; falling back to auto attention")
            self._attention = "auto"
        self._quant_mode = os.getenv("VV_QUANT", "bf16")
        self._diffusion_steps = int(os.getenv("VV_DIFFUSION_STEPS", "5"))
        self._negative_cache_steps = int(os.getenv("VV_NEG_STEPS_CACHE", "2"))
        self._increase_cfg = os.getenv("VV_INCREASE_CFG", "1") == "1"
        self._temperature = float(os.getenv("VV_TEMPERATURE", "0.95"))
        self._free_after_job = os.getenv("VV_FREE_MEMORY_AFTER_JOB", "0") == "1"
        self._use_sampling = os.getenv("VV_USE_SAMPLING", "0") == "1"

        if os.getenv("VV_PRELOAD", "0") == "1":
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        config_key = (
            self._model_handle,
            self._exllama_handle,
            self._attention,
            self._quant_mode,
        )
        if self._model_loaded and config_key == self._active_config:
            return

        logger.info(
            "Loading VibeVoice model %s with exllama %s (attention=%s, quant=%s)",
            self._model_handle,
            self._exllama_handle,
            self._attention,
            self._quant_mode,
        )
        self._node.load_model(
            model_path=self._node._get_model_mapping().get(self._model_handle, self._model_handle),
            attention_type=self._attention,
            quantization_mode=self._quant_mode,
            exllama_model=self._exllama_handle,
        )
        self._model_loaded = True
        self._active_config = config_key

    @staticmethod
    def _load_voice_sample(audio_path: str, target_sr: int = 24000) -> np.ndarray:
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav[:, 0]
        if sr != target_sr:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        wav = wav.astype(np.float32)
        # Treat very short or near-silent references as invalid so we fall back to synthetic
        duration_s = len(wav) / float(sr)
        if duration_s < 0.25 or np.max(np.abs(wav)) < 1e-5:
            raise ValueError("Reference audio too short or silent")
        max_abs = np.max(np.abs(wav))
        if max_abs > 1.0:
            wav = wav / max_abs
        return wav

    def generate(
        self,
        text: str,
        speaker_audio_path: Optional[str],
        cfg_scale: float,
        seed: int,
        top_p: float,
    ) -> str:
        if not text or not text.strip():
            raise ValueError("Text prompt must not be empty")

        self._ensure_model_loaded()

        speakers = ["Speaker 1"]
        voice_samples = []
        if speaker_audio_path:
            try:
                vs = self._load_voice_sample(speaker_audio_path)
                # If the uploaded sample is too short or silent, fall back to synthetic
                if len(vs) < 24000 * 0.25 or float(np.max(np.abs(vs))) < 1e-5:
                    logger.info("Reference audio too short/silent; using synthetic sample")
                    voice_samples = [self._node._create_synthetic_voice_sample(0)]
                else:
                    voice_samples = [vs]
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to load reference audio '%s': %s", speaker_audio_path, exc)
        if not voice_samples:
            voice_samples = [self._node._create_synthetic_voice_sample(0)]
        formatted_text = self._node._format_text_for_vibevoice(text, speakers)
        generation = self._node._generate_with_vibevoice(
            formatted_text,
            voice_samples=voice_samples,
            cfg_scale=cfg_scale,
            seed=seed,
            diffusion_steps=self._diffusion_steps,
            use_sampling=self._use_sampling,
            temperature=self._temperature,
            top_p=top_p,
            streaming=False,
            buffer_duration=1,
            max_new_tokens=32637,
            negative_llm_steps_to_cache=self._negative_cache_steps,
            increase_cfg=self._increase_cfg,
        )

        waveform = generation.get("waveform")
        sample_rate = generation.get("sample_rate", 24000)
        if waveform is None or waveform.numel() == 0:
            raise RuntimeError("VibeVoice returned empty audio")

        audio_np = waveform.squeeze().cpu().numpy()
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        sf.write(temp_path, audio_np, sample_rate)

        if self._free_after_job:
            self._node.free_memory()
            self._model_loaded = False
            self._active_config = None

        return str(temp_path)


_SERVICE = VibeVoiceService()


def _extract_file_path(file_obj) -> Optional[str]:
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path")
    return getattr(file_obj, "name", None)


def _sanitize_seed(val) -> int:
    """Return a safe 32-bit seed. Falls back to 42 on bad input."""
    max_u32 = 2**32 - 1
    try:
        if val is None:
            return 42
        if isinstance(val, str):
            if val.strip() == "":
                return 42
            val = float(val)
        iv = int(val)
    except Exception:
        return 42
    if iv < 0:
        iv = 0
    if iv > max_u32:
        iv = iv % (max_u32 + 1)
    return iv


def generate_audio(
    model,
    text,
    language,
    speaker_audio,
    prefix_audio,
    response_tone_happiness,
    response_tone_sadness,
    response_tone_disgust,
    response_tone_fear,
    response_tone_surprise,
    response_tone_anger,
    response_tone_other,
    response_tone_neutral,
    vq_score,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_overall,
    denoise_speaker,
    cfg_scale,
    top_p,
    min_k,
    min_p,
    linear,
    confidence,
    quadratic,
    seed,
    randomize_seed,
    unconditional_keys,
):
    del (
        model,
        language,
        prefix_audio,
        response_tone_happiness,
        response_tone_sadness,
        response_tone_disgust,
        response_tone_fear,
        response_tone_surprise,
        response_tone_anger,
        response_tone_other,
        response_tone_neutral,
        vq_score,
        fmax,
        pitch_std,
        speaking_rate,
        dnsmos_overall,
        denoise_speaker,
        min_k,
        min_p,
        linear,
        confidence,
        quadratic,
        randomize_seed,
        unconditional_keys,
    )

    if os.getenv("VV_FORCE_SEED", "0") == "1":
        resolved_seed = _sanitize_seed(os.getenv("VV_FIXED_SEED", 42))
    else:
        resolved_seed = _sanitize_seed(seed)

    reference_path = _extract_file_path(speaker_audio)
    cfg_value = 1.4 if cfg_scale in (None, "") else float(cfg_scale)
    top_p_value = 0.95 if top_p in (None, "") else float(top_p)
    return _SERVICE.generate(
        text=text,
        speaker_audio_path=reference_path,
        cfg_scale=cfg_value,
        seed=resolved_seed,
        top_p=top_p_value,
    )


api_inputs = [
    gr.Textbox(label="Model"),
    gr.Textbox(label="Text"),
    gr.Textbox(label="Language"),
    gr.File(label="Speaker Audio"),
    gr.File(label="Prefix Audio"),
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Happiness"),
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Sadness"),
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Disgust"),
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Fear"),
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Surprise"),
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Anger"),
    gr.Slider(minimum=0, maximum=1, value=0.05, label="Other"),
    gr.Slider(minimum=0, maximum=1, value=0.2, label="Neutral"),
    gr.Slider(minimum=0.5, maximum=1.0, value=0.7, label="VQ Score"),
    gr.Slider(minimum=20000, maximum=25000, value=24000, label="Fmax (Hz)"),
    gr.Slider(minimum=20, maximum=150, value=45, label="Pitch Std"),
    gr.Slider(minimum=0, maximum=50, value=14.6, label="Speaking Rate"),
    gr.Slider(minimum=1, maximum=5, value=4, label="DNSMOS Overall"),
    gr.Checkbox(value=True, label="Denoise Speaker"),
    gr.Slider(minimum=1, maximum=10, value=1.4, label="CFG Scale"),
    gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top P"),
    gr.Slider(minimum=1, maximum=100, value=1, label="Min K"),
    gr.Slider(minimum=0.01, maximum=1.0, value=0.2, label="Min P"),
    gr.Checkbox(value=False, label="Linear"),
    gr.Slider(minimum=0, maximum=1, value=0.7, label="Confidence"),
    gr.Checkbox(value=False, label="Quadratic"),
    gr.Number(value=42, label="Seed"),
    gr.Checkbox(value=False, label="Randomize Seed"),
    gr.Textbox(value="[]", label="Unconditional Keys"),
]

app = gr.Interface(
    fn=generate_audio,
    inputs=api_inputs,
    outputs=gr.Audio(label="Generated Audio"),
    title="VibeVoice TTS Zonos-Compatible Wrapper",
    description="Expose VibeVoice TTS with the legacy Zonos HTTP API.",
    api_name="generate_audio",
).queue()


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
