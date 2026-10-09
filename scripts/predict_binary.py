"""Run only the binary crack classifier on one or more photos."""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crack_detection.core import create_model, scan_photo  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--classifier", type=Path, default=ROOT / "models" / "classifier_original.pth")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-hits", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.min_hits < 1:
        parser.error("threshold must be in (0,1); min-hits must be positive")
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=False)
    model = create_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    for photo in args.photos:
        with Image.open(photo) as source:
            image = source.convert("RGB")
        detection = scan_photo(model, image, device,
                               mean=checkpoint["mean"], std=checkpoint["std"],
                               patch_size=checkpoint["img_size"],
                               threshold=args.threshold, min_hits=args.min_hits)
        print(json.dumps({"photo": str(photo.resolve()), **detection}, ensure_ascii=False))


if __name__ == "__main__":
    main()
