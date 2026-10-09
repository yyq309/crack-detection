"""Create mask-verified crack crops; split by original UK masonry photo."""

import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    root = Path(sys.argv[1])
    source = root / "source_data"
    image_zip = zipfile.ZipFile(source / "masonry_crack_images.zip")
    mask_zip = zipfile.ZipFile(source / "masonry_crack_masks.zip")
    image_names = {Path(n).stem: n for n in image_zip.namelist() if n.lower().endswith(".jpg")}
    mask_names = {Path(n).stem.replace("_mask", ""): n for n in mask_zip.namelist()
                  if n.lower().endswith(".png")}
    ids = sorted(image_names.keys() & mask_names.keys())
    if len(ids) != 100:
        raise RuntimeError(f"Expected 100 image/mask pairs, got {len(ids)}")
    output = root / "uk_masonry"
    random_source = random.Random(18458458)
    manifest = []
    for image_id in ids:
        split = "test" if int(image_id[-3:]) % 5 == 0 else "train"
        with Image.open(io.BytesIO(image_zip.read(image_names[image_id]))) as stream:
            image = stream.convert("RGB")
        with Image.open(io.BytesIO(mask_zip.read(mask_names[image_id]))) as stream:
            mask = stream.convert("L")
        if image.size != mask.size:
            raise RuntimeError(f"Size mismatch for {image_id}: {image.size} vs {mask.size}")
        scale = min(1.0, 1536 / max(image.size))
        size = tuple(max(224, round(v * scale)) for v in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
        mask = mask.resize(size, Image.Resampling.NEAREST)
        binary = np.array(mask) > 127
        yy, xx = np.where(binary)
        if len(xx) < 50:
            print("skip no substantial mask", image_id, len(xx), flush=True)
            continue
        width, height = image.size
        positions = []
        max_crops = 12 if split == "train" else 8
        for _ in range(300):
            k = random_source.randrange(len(xx))
            left = min(max(0, int(xx[k]) - 112 + random_source.randint(-72, 72)), width - 224)
            top = min(max(0, int(yy[k]) - 112 + random_source.randint(-72, 72)), height - 224)
            if any(abs(left - x) + abs(top - y) < 80 for x, y in positions):
                continue
            crop_mask = binary[top:top + 224, left:left + 224]
            cy, cx = np.where(crop_mask)
            if len(cx) < 35 or max(np.ptp(cx), np.ptp(cy)) < 30:
                continue
            positions.append((left, top))
            path = output / split / "crack" / f"{image_id}_{len(positions):02d}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.crop((left, top, left + 224, top + 224)).save(path, quality=92)
            manifest.append({"file": str(path.relative_to(root)), "source_image": image_id,
                             "split": split, "left": left, "top": top,
                             "mask_pixels": len(cx)})
            if len(positions) >= max_crops:
                break
        print(image_id, split, len(positions), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("crops", len(manifest), "train", sum(x["split"] == "train" for x in manifest),
          "test", sum(x["split"] == "test" for x in manifest), flush=True)


if __name__ == "__main__":
    main()
