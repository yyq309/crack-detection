"""Mine high-scoring false crack windows from known joint-only training photos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from run_experiment import MEAN, STD, create_model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--per-photo", type=int, default=80)
    parser.add_argument("--min-score", type=float, default=0.3)
    args = parser.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_model(ck).cuda().eval()
    normalize = transforms.Normalize(MEAN, STD)
    out = args.data / "mined_local"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for path in sorted((args.data / "local_joints").glob("*.jpg")):
        with Image.open(path) as source:
            original = source.convert("RGB")
        candidates = []
        w0, h0 = original.size
        for long_side in (512, 768, 1024):
            scale = min(1.0, long_side / max(w0, h0))
            w, h = round(w0 * scale), round(h0 * scale)
            resized = original.resize((w, h))
            xs = list(range(0, max(1, w - 224 + 1), 64)) or [0]
            ys = list(range(0, max(1, h - 224 + 1), 64)) or [0]
            if w >= 224 and xs[-1] != w - 224:
                xs.append(w - 224)
            if h >= 224 and ys[-1] != h - 224:
                ys.append(h - 224)
            patches = []
            coords = []
            for y in ys:
                for x in xs:
                    patch = resized.crop((x, y, x + 224, y + 224))
                    if patch.size != (224, 224):
                        patch = patch.resize((224, 224))
                    patches.append(patch)
                    coords.append((x, y))
            for start in range(0, len(patches), 128):
                batch = torch.stack([normalize(transforms.functional.to_tensor(im))
                                     for im in patches[start:start + 128]]).cuda()
                with torch.autocast("cuda"):
                    probability = torch.softmax(model(batch).float(), dim=1)[:, 0].cpu().tolist()
                for patch, (x, y), p in zip(patches[start:start + 128], coords[start:start + 128], probability):
                    if p >= args.min_score:
                        candidates.append((p, long_side, x, y, patch))

        # Suppress near-duplicate overlapping windows within each scale.
        selected = []
        for candidate in sorted(candidates, key=lambda x: x[0], reverse=True):
            p, scale, x, y, _ = candidate
            if any(scale == q[1] and abs(x - q[2]) < 112 and abs(y - q[3]) < 112 for q in selected):
                continue
            selected.append(candidate)
            if len(selected) >= args.per_photo:
                break
        for rank, (p, scale, x, y, patch) in enumerate(selected):
            patch.save(out / f"{path.stem}_{rank:03d}_{scale}_{x}_{y}.jpg", quality=95)
        report[path.name] = {"candidate_windows": len(candidates), "selected": len(selected),
                             "max_probability": round(max([c[0] for c in candidates], default=0), 5)}
        print(path.name, report[path.name], flush=True)
    (out / "mining_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
