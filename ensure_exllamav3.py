#!/usr/bin/env python3
"""Exit 0 if exllamav3 0.0.6 is ready, non-zero otherwise."""

import sys


def main() -> int:
    try:
        import exllamav3  # type: ignore
    except Exception:
        return 1

    version = getattr(exllamav3, "__version__", "0.0.0")
    return 0 if version == "0.0.6" else 1


if __name__ == "__main__":
    sys.exit(main())
