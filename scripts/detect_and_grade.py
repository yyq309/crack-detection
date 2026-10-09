"""Command-line entry point for binary detection followed by crack grading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crack_detection.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
