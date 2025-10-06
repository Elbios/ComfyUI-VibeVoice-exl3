#!/usr/bin/env python
"""Preload VibeVoice weights so the first request is instantaneous."""
from __future__ import annotations

import os
import sys
import traceback


def main() -> int:
    print(
        f"[prefetch] Preparing VibeVoice model={os.getenv('VV_MODEL')} "
        f"exllama={os.getenv('VV_EXLLAMA_MODEL')} attention={os.getenv('VV_ATTENTION')}"
    )

    try:
        from VIBEVOICE_ZONOS_WRAPPER import VibeVoiceService  # pylint: disable=import-error
    except Exception as exc:  # pragma: no cover - log and exit
        print(f"[prefetch] Wrapper import failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    service = VibeVoiceService()
    try:
        service._ensure_model_loaded()  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - log and exit
        print(f"[prefetch] Model preload failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print("[prefetch] VibeVoice weights downloaded and ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
