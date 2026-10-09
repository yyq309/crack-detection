"""Train a three-class crack/background/constructed-joint ResNet18."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

from run_experiment import (ImageRecords, Record, PHOTO_POSITIVE, classification_metrics,
                            paths, predict_records, scan_photo)
from PIL import Image


class ThreeClassDataset(Dataset):
    def __init__(self, records: list[Record]):
        self.records = records
        self.images = ImageRecords(records, training=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image, binary_label = self.images[index]
        source = self.records[index].source
        target = 0 if binary_label == 1 else (1 if source == "base_nocrack" else 2)
        return image, target


def three_class_model(base: dict) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 3))
    state = {k: v for k, v in base["state_dict"].items() if not k.startswith("fc.1.")}
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        model.fc[1].weight[:2].copy_(base["state_dict"]["fc.1.weight"])
        model.fc[1].bias[:2].copy_(base["state_dict"]["fc.1.bias"])
        model.fc[1].weight[2].copy_(base["state_dict"]["fc.1.weight"][1])
        model.fc[1].bias[2].copy_(base["state_dict"]["fc.1.bias"][1])
    return model


def collect(root: Path):
    train = []
    for cls, label in (("crack", 1), ("nocrack", 0)):
        train += paths(root, f"base/train/{cls}/*.jpg", label, f"base_{cls}")
        brick = paths(root, f"brickwork/{cls}/*.jpg", label, f"brick_{cls}")
        train += [r for r in brick if int(r.path.stem) % 5 != 0]
        train += paths(root, f"crack_ch/naillac/{cls}/*.jpg", label, f"stone_{cls}")
    train += paths(root, "local_joints/*.jpg", 0, "local_joint")
    train += paths(root, "mined_local/*.jpg", 0, "mined_joint")
    train += [r for r in paths(root, "chambord/train/*.png", 0, "chambord")
              if int(r.path.stem) % 10 != 0]
    val_base = sum((paths(root, f"base/val/{cls}/*.jpg", label, f"base_{cls}")
                    for cls, label in (("crack", 1), ("nocrack", 0))), [])
    val_joint = [r for r in paths(root, "chambord/train/*.png", 0, "chambord")
                 if int(r.path.stem) % 10 == 0]
    val_stone = sum((paths(root, f"crack_ch/saint_nikolaos/{cls}/*.jpg", label, f"stone_{cls}")
                     for cls, label in (("crack", 1), ("nocrack", 0))), [])
    brick_test = sum(([r for r in paths(root, f"brickwork/{cls}/*.jpg", label, f"brick_{cls}")
                       if int(r.path.stem) % 5 == 0]
                      for cls, label in (("crack", 1), ("nocrack", 0))), [])
    tests = {
        "base_test": sum((paths(root, f"base/test/{cls}/*.jpg", label, f"base_{cls}")
                          for cls, label in (("crack", 1), ("nocrack", 0))), []),
        "brick_test": brick_test,
        "chambord_test": paths(root, "chambord/test/*.png", 0, "chambord"),
        "stone_site_validation": val_stone,
    }
    return train, val_base, val_joint, val_stone, tests


def field_eval(model: nn.Module, root: Path, device: torch.device):
    predictions = {}
    for path in sorted((root / "real_holdout").glob("*.jpg")):
        with Image.open(path) as stream:
            predictions[path.name] = scan_photo(model, stream.convert("RGB"), device)
        print(path.name, predictions[path.name], flush=True)
    labels = [int(n in PHOTO_POSITIVE) for n in predictions]
    result = classification_metrics(labels, [float(predictions[n]["predicted_crack"]) for n in predictions])
    return {"manual_labels": result, "predictions": predictions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = torch.load(args.data / "models/baseline.pth", map_location="cpu", weights_only=False)
    model = three_class_model(baseline).to(device)
    train, val_base, val_joint, val_stone, tests = collect(args.data)
    counts = Counter(r.source for r in train)
    print("train", dict(counts), "tests", {k: len(v) for k, v in tests.items()}, flush=True)
    shares = {"base_crack": .28, "base_nocrack": .12, "brick_crack": .15,
              "brick_nocrack": .10, "stone_crack": .12, "stone_nocrack": .06,
              "chambord": .09, "local_joint": .01, "mined_joint": .07}
    sampler = WeightedRandomSampler([shares[r.source] / counts[r.source] for r in train],
                                    num_samples=6000, replacement=True)
    loader = DataLoader(ThreeClassDataset(train), batch_size=args.batch, sampler=sampler,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best_score = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        loss_total = 0.0
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.item() * len(images)
        scheduler.step()
        b = predict_records(model, val_base, device, args.batch)
        j = predict_records(model, val_joint, device, args.batch)
        s = predict_records(model, val_stone, device, args.batch)
        score = .4 * b["balanced_accuracy"] + .45 * s["balanced_accuracy"] + .15 * j["specificity"]
        row = {"epoch": epoch, "loss": round(loss_total / 6000, 5), "base_val": b,
               "stone_site_val": s, "joint_val": j, "selection_score": round(score, 5),
               "seconds": round(time.time() - start, 1)}
        history.append(row)
        print("epoch", json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "classes": ["crack", "nocrack", "joint"],
                        "mean": baseline["mean"], "std": baseline["std"], "img_size": 224,
                        "source_shares": shares, "epoch": epoch}, args.output / "best_model_3class.pth")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    ck = torch.load(args.output / "best_model_3class.pth", map_location="cpu", weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(model.fc.in_features, 3))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    result = {k: predict_records(model, v, device, args.batch) for k, v in tests.items()}
    result["field_23"] = field_eval(model, args.data, device)
    (args.output / "final_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("final", {k: v for k, v in result.items() if k != "field_23"},
          "field", result["field_23"]["manual_labels"], flush=True)


if __name__ == "__main__":
    main()
