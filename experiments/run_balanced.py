"""Balance mask-verified masonry cracks against public and local joint negatives."""

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
from torch.utils.data import DataLoader, WeightedRandomSampler

from run_experiment import (ImageRecords, build_records, create_model, evaluate_all,
                            paths, predict_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    torch.cuda.manual_seed_all(44)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = torch.load(args.data / "models/baseline.pth", map_location="cpu", weights_only=False)
    start = torch.load(args.start, map_location="cpu", weights_only=False) if args.start else baseline
    model = create_model(start).to(device)
    train, val_base, val_joint, tests = build_records(args.data)
    uk_train = paths(args.data, "uk_masonry/train/crack/*.jpg", 1, "uk_crack")
    uk_test = paths(args.data, "uk_masonry/test/crack/*.jpg", 1, "uk_crack")
    if len(uk_train) < 500 or len(uk_test) < 100:
        raise RuntimeError(f"Missing UK masonry crops: {len(uk_train)}, {len(uk_test)}")
    train += uk_train
    tests["uk_masonry_unseen_photos"] = uk_test
    counts = Counter(r.source for r in train)
    print("train", dict(counts), "tests", {k: len(v) for k, v in tests.items()}, flush=True)

    shares = {
        "base_crack": .26, "base_nocrack": .10,
        "uk_crack": .21,
        "brick_crack": .12, "brick_nocrack": .06,
        "stone_crack": .06, "stone_nocrack": .05,
        "chambord": .06, "local_joint": .01, "mined_joint": .07,
    }
    assert round(sum(shares.values()), 6) == 1.0
    sampler = WeightedRandomSampler([shares[r.source] / counts[r.source] for r in train],
                                    num_samples=args.samples, replacement=True)
    loader = DataLoader(ImageRecords(train, True), batch_size=args.batch, sampler=sampler,
                        num_workers=4, pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    history, best_score = [], -1.0
    val_stone = tests.pop("stone_site_holdout")
    tests["stone_site_holdout"] = val_stone
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_total = 0.0
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), (1 - labels).to(device, non_blocking=True)
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
        score = .4 * b["balanced_accuracy"] + .4 * s["balanced_accuracy"] + .2 * j["specificity"]
        row = {"epoch": epoch, "loss": round(loss_total / args.samples, 5),
               "base_val": b, "joint_val": j, "stone_site_val": s,
               "selection_score": round(score, 5), "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print("epoch", json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            torch.save({**baseline, "state_dict": model.state_dict(),
                        "balanced_finetune": {"epoch": epoch, "source_shares": shares,
                                              "selection_score": score}},
                       args.output / "best_model_balanced.pth")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best = torch.load(args.output / "best_model_balanced.pth", map_location="cpu", weights_only=False)
    model = create_model(best).to(device)
    result = evaluate_all(model, args.data, device, args.batch, tests)
    result["base_val"] = predict_records(model, val_base, device, args.batch)
    result["joint_val"] = predict_records(model, val_joint, device, args.batch)
    (args.output / "final_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("final", {k: v for k, v in result.items() if k != "real_23_predictions"}, flush=True)


if __name__ == "__main__":
    main()
