"""Create crack-positive training crops from the DamCrack dam segmentation archive.

DamCrack contains crack (red) and spalling (blue) annotations. The archive's
spalling-only and unannotated images are inventoried, but never automatically
called crack-free: small cracks can be missed by an annotation. All output is
training-only because the released patches lack original capture/group IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ARCHIVE_MD5 = "f193ad7c45c7a7ee31786562513f76ce"
ROOT = "Damage Segmentaion/"
WINDOW = 224
STRIDE = 56


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidates(red: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (crack pixels, x, y) for all 224 px grid windows."""
    h, w = red.shape
    integral = np.pad(red.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    xs = sorted(set(range(0, w - WINDOW + 1, STRIDE)) | {w - WINDOW})
    ys = sorted(set(range(0, h - WINDOW + 1, STRIDE)) | {h - WINDOW})
    out = []
    for y in ys:
        for x in xs:
            n = int(integral[y + WINDOW, x + WINDOW] - integral[y, x + WINDOW]
                    - integral[y + WINDOW, x] + integral[y, x])
            out.append((n, x, y))
    return sorted(out, reverse=True)


def select_windows(red: np.ndarray, max_crops: int, min_pixels: int) -> list[tuple[int, int, int]]:
    selected = []
    for count, x, y in candidates(red):
        if count < min_pixels:
            break
        if any(max(0, WINDOW - abs(x - px)) * max(0, WINDOW - abs(y - py))
               > WINDOW * WINDOW * 0.65 for _, px, py in selected):
            continue
        selected.append((count, x, y))
        if len(selected) == max_crops:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-crops", type=int, default=2)
    parser.add_argument("--min-crack-pixels", type=int, default=100)
    parser.add_argument("--skip-checksum", action="store_true")
    args = parser.parse_args()
    if args.max_crops < 1 or args.min_crack_pixels < 1:
        parser.error("crop and pixel limits must be positive")
    actual_md5 = md5_file(args.archive)
    if not args.skip_checksum and actual_md5 != ARCHIVE_MD5:
        raise ValueError(f"Archive MD5 mismatch: {actual_md5}")

    out_images = args.output / "train" / "crack"
    out_images.mkdir(parents=True, exist_ok=True)
    manifest = []
    inventory = Counter()
    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        image_paths = sorted(n for n in names if n.startswith(ROOT + "Images/")
                             and n.lower().endswith(".jpg"))
        for image_path in image_paths:
            stem = Path(image_path).stem
            mask_path = ROOT + f"Refined Labels/Mask/{stem}.png"
            if mask_path not in names:
                raise FileNotFoundError(mask_path)
            with Image.open(io.BytesIO(archive.read(mask_path))) as source:
                mask = np.asarray(source.convert("RGB"))
            valid = (((mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0))
                     | ((mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0))
                     | ((mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)))
            if not valid.all():
                raise ValueError(f"Unexpected mask colors in {stem}")
            red = (mask[:, :, 0] == 255)
            blue = (mask[:, :, 2] == 255)
            kind = "crack_and_spalling" if red.any() and blue.any() else (
                "crack_only" if red.any() else "spalling_only" if blue.any() else "unannotated")
            inventory[kind] += 1
            windows = select_windows(red, args.max_crops, args.min_crack_pixels)
            if not windows:
                continue
            with Image.open(io.BytesIO(archive.read(image_path))) as source:
                photo = source.convert("RGB")
            if photo.size != (640, 640) or mask.shape[:2] != (640, 640):
                raise ValueError(f"Unexpected image/mask dimensions for {stem}")
            for index, (count, x, y) in enumerate(windows):
                name = f"{stem}_{index:02d}.jpg"
                photo.crop((x, y, x + WINDOW, y + WINDOW)).save(
                    out_images / name, quality=93, subsampling=0)
                manifest.append({"file": f"train/crack/{name}", "label": "crack",
                                 "source_image": stem, "source_archive_path": image_path,
                                 "x": x, "y": y, "width": WINDOW, "height": WINDOW,
                                 "crack_pixels": count, "image_annotation": kind})
    expected = {Path(item["file"]).name for item in manifest}
    stale = {p.name for p in out_images.glob("*.jpg")} - expected
    if stale:
        raise ValueError(f"Output contains {len(stale)} stale crops; use a fresh output directory")
    metadata = {
        "source": "DamCrack", "record": "https://zenodo.org/records/17274707",
        "license": "CC BY 4.0; credit Gharehbaghi et al. and the DamCrack dataset",
        "archive_md5": actual_md5, "mask_version": "Refined Labels",
        "mask_colors": {"red": "crack", "blue": "spalling", "black": "unannotated"},
        "use": "training-only crack-positive crops; no automatic negatives or independent dam test",
        "image_inventory": dict(inventory), "crop_count": len(manifest),
        "selection": {"window": WINDOW, "stride": STRIDE, "max_crops": args.max_crops,
                      "min_crack_pixels": args.min_crack_pixels},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
