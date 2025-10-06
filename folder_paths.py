"""Minimal ComfyUI folder_paths shim for standalone use.

This module only exposes the subset of APIs used by the embedded VibeVoice
nodes so they can run outside of ComfyUI. Paths are localized under the
repository's directory structure.
"""
from __future__ import annotations

import os
from pathlib import Path

# Base directory for all generated folders (repo root by default)
_BASE_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _BASE_DIR / "models"
_CHECKPOINTS_DIR = _MODELS_DIR / "checkpoints"
_INPUT_DIR = _BASE_DIR / "inputs"
_OUTPUT_DIR = _BASE_DIR / "outputs"
_TEMP_DIR = _BASE_DIR / "temp"

for _path in (_MODELS_DIR, _CHECKPOINTS_DIR, _INPUT_DIR, _OUTPUT_DIR, _TEMP_DIR):
    _path.mkdir(parents=True, exist_ok=True)


def get_folder_paths(category: str):
    """Replicate ComfyUI's folder lookup semantics for checkpoints.

    Currently only the "checkpoints" category is required by BaseVibeVoiceNode.
    """
    if category != "checkpoints":
        raise ValueError(f"Unsupported folder category: {category}")
    return [str(_CHECKPOINTS_DIR)]


def get_input_directory() -> str:
    return str(_INPUT_DIR)


def get_output_directory() -> str:
    return str(_OUTPUT_DIR)


def get_temp_directory() -> str:
    return str(_TEMP_DIR)
