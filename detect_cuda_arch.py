#!/usr/bin/env python3
"""Print the primary GPU's compute capability, if available."""

import sys


def main() -> int:
    try:
        import torch
    except Exception:
        return 0

    if not torch.cuda.is_available():
        return 0

    maj, min_ = torch.cuda.get_device_capability(0)
    print(f"{maj}.{min_}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
