"""Binary crack detection followed by the handoff segmentation and grading rules.

The handoff checkpoints and grading modules are read only. All output goes into
this experiment directory unless --output is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from .core import create_model, scan_photo
from .quantify import quantify
from .grade import grade


HERE = Path(__file__).resolve().parents[1]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def block(in_channels: int, out_channels: int) -> nn.Module:
    # Architecture from the handoff train_seg.py, needed to load its checkpoint.
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.BatchNorm2d(out_channels), nn.ReLU(True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1),
                         nn.BatchNorm2d(out_channels), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3, self.e4 = (block(3, b), block(b, b * 2),
                                               block(b * 2, b * 4), block(b * 4, b * 8))
        self.pool = nn.MaxPool2d(2)
        self.bott = block(b * 8, b * 16)
        self.u4, self.d4 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2), block(b * 16, b * 8)
        self.u3, self.d3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2), block(b * 8, b * 4)
        self.u2, self.d2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2), block(b * 4, b * 2)
        self.u1, self.d1 = nn.ConvTranspose2d(b * 2, b, 2, 2), block(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        z = self.bott(self.pool(e4))
        d = self.d4(torch.cat([self.u4(z), e4], 1))
        d = self.d3(torch.cat([self.u3(d), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        return self.out(d)


def load_models(classifier_path: Path, segmenter_path: Path, device: torch.device):
    ck = torch.load(classifier_path, map_location="cpu", weights_only=False)
    if ck.get("classes") != ["crack", "nocrack"] or ck.get("img_size") != 224:
        raise ValueError("Classifier must use the handoff crack/nocrack 224px checkpoint format")
    classifier = create_model(ck).to(device).eval()
    seg_ck = torch.load(segmenter_path, map_location="cpu", weights_only=False)
    segmenter = UNet().to(device)
    segmenter.load_state_dict(seg_ck["state_dict"])
    segmenter.eval()
    return classifier, segmenter, ck


def resize_for_grade(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if scale == 1:
        return image
    return cv2.resize(image, (max(16, round(w * scale)), max(16, round(h * scale))))


@torch.no_grad()
def segment_photo(model: UNet, image: np.ndarray, device: torch.device,
                  long_side: int = 480, win: int = 192, stride: int = 96) -> np.ndarray:
    """Match the handoff overlapping U-Net windows and their mean fusion."""
    image = resize_for_grade(image, long_side)
    h, w = image.shape[:2]
    if h < win or w < win:
        positions = [(0, 0)]
        tiles = [image]
    else:
        ys = sorted(set(list(range(0, h - win + 1, stride)) + [h - win]))
        xs = sorted(set(list(range(0, w - win + 1, stride)) + [w - win]))
        positions = [(y, x) for y in ys for x in xs]
        tiles = [image[y:y + win, x:x + win] for y, x in positions]
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for start in range(0, len(tiles), 32):
        batch = tiles[start:start + 32]
        th, tw = batch[0].shape[:2]
        ph, pw = (-th) % 16, (-tw) % 16
        arrays = [np.pad(t, ((0, ph), (0, pw), (0, 0)), mode="reflect") for t in batch]
        x = np.stack([((a.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
                      for a in arrays])
        probs = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()[:, 0, :th, :tw]
        for (y, x0), p in zip(positions[start:start + 32], probs):
            acc[y:y + th, x0:x0 + tw] += p
            cnt[y:y + th, x0:x0 + tw] += 1
    return acc / np.maximum(cnt, 1)


def output_image(image: np.ndarray, mask: np.ndarray, target: Path):
    shown = resize_for_grade(image, max(mask.shape))
    shown = shown.copy()
    shown[mask] = (0.35 * shown[mask] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
    Image.fromarray(shown).save(target)


def analyze_photo(photo: Path, classifier: nn.Module, segmenter: UNet, classifier_checkpoint: dict,
                  device: torch.device, args, output: Path) -> dict:
    with Image.open(photo) as stream:
        image = stream.convert("RGB")
    detection = scan_photo(classifier, image, device,
                           mean=classifier_checkpoint["mean"], std=classifier_checkpoint["std"],
                           patch_size=classifier_checkpoint["img_size"],
                           threshold=args.class_threshold,
                           min_hits=args.min_hits)
    result = {"photo": str(photo.resolve()), "crack_detected": detection["predicted_crack"],
              "detection": detection, "grade_status": "no_crack", "grade": None,
              "measurements": None, "mask_overlay": None}
    if not detection["predicted_crack"]:
        return result
    rgb = np.asarray(image)
    working = resize_for_grade(rgb, args.work_long)
    probability = segment_photo(segmenter, rgb, device, long_side=args.work_long)
    mask = probability >= args.seg_threshold
    measurements = quantify(mask, mm_per_px=args.mm_per_px)
    if measurements["area_px"] == 0:
        result["grade_status"] = "ungradable_no_mask"
        result["measurements"] = measurements
        return result
    grading = grade(measurements, dam_mode=args.dam_mode,
                    dam_axis_deg=args.dam_axis_deg)
    target = output / f"{photo.stem}_mask_overlay.png"
    output_image(working, mask, target)
    result.update({"grade_status": "graded", "grade": grading,
                   "measurements": measurements, "mask_overlay": str(target.resolve())})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--classifier", type=Path, default=HERE / "models" / "classifier_original.pth")
    parser.add_argument("--segmenter", type=Path, default=HERE / "models" / "segmenter_original.pth")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    parser.add_argument("--class-threshold", type=float, default=0.6)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--seg-threshold", type=float, default=0.5)
    parser.add_argument("--work-long", type=int, default=480)
    parser.add_argument("--mm-per-px", type=float, help="Camera calibration at the working image scale")
    parser.add_argument("--dam-mode", action="store_true")
    parser.add_argument("--dam-axis-deg", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 < args.class_threshold < 1 or not 0 < args.seg_threshold < 1:
        parser.error("Probability thresholds must be between 0 and 1")
    if args.min_hits < 1 or args.work_long < 224 or (args.mm_per_px is not None and args.mm_per_px <= 0):
        parser.error("Invalid min-hits, work-long, or mm-per-px")
    for photo in args.photos:
        if not photo.is_file():
            parser.error(f"Image does not exist: {photo}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, segmenter, checkpoint = load_models(args.classifier, args.segmenter, device)
    results = [analyze_photo(photo, classifier, segmenter, checkpoint, device, args, args.output)
               for photo in args.photos]
    destination = args.output / "results.json"
    destination.write_text(json.dumps({"classifier": str(args.classifier.resolve()),
                                       "segmenter": str(args.segmenter.resolve()),
                                       "calibrated": args.mm_per_px is not None,
                                       "results": results}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    for result in results:
        level = result["grade"]["level"] if result["grade"] else None
        print(json.dumps({"photo": Path(result["photo"]).name,
                          "crack_detected": result["crack_detected"],
                          "grade_status": result["grade_status"], "level": level},
                         ensure_ascii=False))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
