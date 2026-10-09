"""Fine-tune the handoff ResNet18 with masonry and joint hard negatives.

Run from an extracted training_bundle.zip on a CUDA machine. The original
checkpoint and the 23-photo field set are never changed or used for training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PHOTO_POSITIVE = {
    "C_wall_1.jpg", "retw_1.jpg", "retw_2.jpg", "road_1.jpg", "walk_2.jpg",
    "C_pave_1.jpg", "C_road_1.jpg", "brdg_2.jpg", "cpav_1.jpg", "ncpav_1.jpg",
    "plas_1.jpg", "walk_1.jpg", "C_brick_1.jpg", "cflr_2.jpg",
}


@dataclass(frozen=True)
class Record:
    path: Path
    label: int  # crack=1
    source: str


class ImageRecords(Dataset):
    def __init__(self, records: list[Record], training: bool):
        self.records = records
        self.training = training
        self.standard = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.78, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
        self.large = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.22, 0.65), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
        self.mask_cropped_positive = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
        self.eval_tf = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source:
            img = source.convert("RGB")
        if self.training and record.source == "local_joint":
            tf = self.large
        elif self.training and record.source == "uk_crack":
            tf = self.mask_cropped_positive
        else:
            tf = self.standard if self.training else self.eval_tf
        return tf(img), record.label


def paths(root: Path, pattern: str, label: int, source: str) -> list[Record]:
    return [Record(path, label, source) for path in sorted(root.glob(pattern))]


def build_records(root: Path):
    train = []
    for cls, label in (("crack", 1), ("nocrack", 0)):
        train += paths(root, f"base/train/{cls}/*.jpg", label, f"base_{cls}")
        train += paths(root, f"brickwork/{cls}/*.jpg", label, f"brick_{cls}")
        train += paths(root, f"crack_ch/naillac/{cls}/*.jpg", label, f"stone_{cls}")
    train += paths(root, "local_joints/*.jpg", 0, "local_joint")
    train += paths(root, "mined_local/*.jpg", 0, "mined_joint")
    chambord = paths(root, "chambord/train/*.png", 0, "chambord")
    train += [r for r in chambord if int(r.path.stem) % 10 != 0]
    val_joint = [r for r in chambord if int(r.path.stem) % 10 == 0]
    val_base = sum((paths(root, f"base/val/{cls}/*.jpg", label, f"base_{cls}")
                    for cls, label in (("crack", 1), ("nocrack", 0))), [])
    tests = {
        "base_test": sum((paths(root, f"base/test/{cls}/*.jpg", label, f"base_{cls}")
                          for cls, label in (("crack", 1), ("nocrack", 0))), []),
        "stone_site_holdout": sum((paths(root, f"crack_ch/saint_nikolaos/{cls}/*.jpg", label, f"stone_{cls}")
                                   for cls, label in (("crack", 1), ("nocrack", 0))), []),
        "chambord_test": paths(root, "chambord/test/*.png", 0, "chambord"),
    }
    return train, val_base, val_joint, tests


def create_model(checkpoint: dict) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def predict_records(model: nn.Module, records: list[Record], device: torch.device, batch: int) -> dict:
    model.eval()
    loader = DataLoader(ImageRecords(records, False), batch_size=batch, shuffle=False, num_workers=0)
    probs = []
    for images, _ in loader:
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        probs.extend(torch.softmax(logits.float(), dim=1)[:, 0].cpu().tolist())
    labels = [r.label for r in records]
    return classification_metrics(labels, probs)


def classification_metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict:
    tp = sum(y == 1 and p >= threshold for y, p in zip(labels, probs))
    fn = sum(y == 1 and p < threshold for y, p in zip(labels, probs))
    tn = sum(y == 0 and p < threshold for y, p in zip(labels, probs))
    fp = sum(y == 0 and p >= threshold for y, p in zip(labels, probs))
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return {"n": len(labels), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "recall": round(rec, 4), "specificity": round(spec, 4),
            "balanced_accuracy": round((rec + spec) / 2, 4)}


@torch.no_grad()
def scan_photo(model: nn.Module, image: Image.Image, device: torch.device,
               scales=(512, 768, 1024), stride=64, threshold=0.6, min_hits=2) -> dict:
    """Match the handoff multi-scale photo decision, including overlap and thresholds."""
    model.eval()
    best = None
    normalize = transforms.Normalize(MEAN, STD)
    w0, h0 = image.size
    for long_side in scales:
        scale = min(1.0, long_side / max(w0, h0))
        w, h = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        resized = image.resize((w, h))
        xs = list(range(0, max(1, w - 224 + 1), stride)) or [0]
        ys = list(range(0, max(1, h - 224 + 1), stride)) or [0]
        if w >= 224 and xs[-1] != w - 224:
            xs.append(w - 224)
        if h >= 224 and ys[-1] != h - 224:
            ys.append(h - 224)
        patches = []
        for y in ys:
            for x in xs:
                patch = resized.crop((x, y, x + 224, y + 224))
                if patch.size != (224, 224):
                    patch = patch.resize((224, 224))
                patches.append(normalize(transforms.functional.to_tensor(patch)))
        probabilities = []
        for start in range(0, len(patches), 128):
            batch = torch.stack(patches[start:start + 128]).to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(batch)
            probabilities.extend(torch.softmax(logits.float(), dim=1)[:, 0].cpu().tolist())
        hits = sum(p >= threshold for p in probabilities)
        peak = max(probabilities, default=0.0)
        candidate = {"hits": hits, "peak": round(peak, 5), "windows": len(probabilities),
                     "scale": long_side, "predicted_crack": hits >= min_hits}
        if best is None or (hits, peak) > (best["hits"], best["peak"]):
            best = candidate
    return best


def evaluate_all(model: nn.Module, root: Path, device: torch.device, batch: int, tests: dict) -> dict:
    result = {name: predict_records(model, records, device, batch) for name, records in tests.items()}
    real = {}
    for path in sorted((root / "real_holdout").glob("*.jpg")):
        with Image.open(path) as source:
            image = source.convert("RGB")
        real[path.name] = scan_photo(model, image, device)
        print(f"  {path.name:18s} crack={real[path.name]['predicted_crack']} "
              f"hits={real[path.name]['hits']}", flush=True)
    if len(real) != 23:
        raise ValueError(f"Expected 23 field photos, got {len(real)}")
    labels = [int(name in PHOTO_POSITIVE) for name in real]
    result["real_23_manual"] = classification_metrics(
        labels, [float(real[name]["predicted_crack"]) for name in real]
    )
    intent_positive = PHOTO_POSITIVE - {"ncpav_1.jpg"}
    result["real_23_manifest_intent"] = classification_metrics(
        [int(name in intent_positive) for name in real],
        [float(real[name]["predicted_crack"]) for name in real]
    )
    result["real_23_predictions"] = real
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epoch-samples", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--start-checkpoint", type=Path)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.start_checkpoint or args.data / "models/baseline.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["classes"] != ["crack", "nocrack"]:
        raise ValueError(f"Unexpected class order: {checkpoint['classes']}")
    model = create_model(checkpoint).to(device)
    train, val_base, val_joint, tests = build_records(args.data)
    by_source = defaultdict(int)
    for r in train:
        by_source[r.source] += 1
    print("device", device, "train_sources", dict(by_source), "tests",
          {k: len(v) for k, v in tests.items()}, flush=True)

    if not args.skip_baseline:
        print("baseline evaluation", flush=True)
        baseline = evaluate_all(model, args.data, device, args.batch, tests)
        baseline["base_val"] = predict_records(model, val_base, device, args.batch)
        baseline["joint_val"] = predict_records(model, val_joint, device, args.batch)
        (args.output / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print("baseline summary", {k: v for k, v in baseline.items() if k != "real_23_predictions"}, flush=True)

    # Fixed source mixture prevents abundant road patches from drowning out stone joints.
    if by_source["mined_joint"]:
        source_share = {
            "base_crack": .30, "base_nocrack": .11, "brick_crack": .12,
            "brick_nocrack": .10, "stone_crack": .08, "stone_nocrack": .06,
            "chambord": .10, "local_joint": .01, "mined_joint": .12,
        }
    else:
        source_share = {
            "base_crack": .30, "base_nocrack": .15, "brick_crack": .10,
            "brick_nocrack": .13, "stone_crack": .05, "stone_nocrack": .07,
            "chambord": .18, "local_joint": .02,
        }
    weights = [source_share[r.source] / by_source[r.source] for r in train]
    sampler = WeightedRandomSampler(weights, num_samples=args.epoch_samples, replacement=True)
    loader = DataLoader(ImageRecords(train, True), batch_size=args.batch, sampler=sampler,
                        num_workers=4, pin_memory=device.type == "cuda", persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = (1 - labels).to(device, non_blocking=True)  # checkpoint index 0 = crack
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(images)
        scheduler.step()
        b = predict_records(model, val_base, device, args.batch)
        j = predict_records(model, val_joint, device, args.batch)
        score = (b["balanced_accuracy"] + j["specificity"]) / 2
        row = {"epoch": epoch, "loss": round(total_loss / args.epoch_samples, 5),
               "base_val": b, "joint_val": j, "selection_score": round(score, 5),
               "seconds": round(time.time() - start, 1)}
        history.append(row)
        print("epoch", json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            torch.save({**checkpoint, "state_dict": model.state_dict(),
                        "hard_negative_finetune": {"epoch": epoch, "source_share": source_share,
                                                   "selection_score": score}},
                       args.output / "best_model_hardneg.pth")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    best = torch.load(args.output / "best_model_hardneg.pth", map_location="cpu", weights_only=False)
    model = create_model(best).to(device)
    print("final held-out evaluation", flush=True)
    final = evaluate_all(model, args.data, device, args.batch, tests)
    final["base_val"] = predict_records(model, val_base, device, args.batch)
    final["joint_val"] = predict_records(model, val_joint, device, args.batch)
    (args.output / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("final summary", {k: v for k, v in final.items() if k != "real_23_predictions"}, flush=True)


if __name__ == "__main__":
    main()
