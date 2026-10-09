"""Inspect full-frame evidence without changing the held-out decisions."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from run_experiment import MEAN, STD, PHOTO_POSITIVE, create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, path in (("baseline", args.data / "models/baseline.pth"), ("finetuned", args.finetuned)):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        models[name] = create_model(ck).to(device).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    print("name,truth_manual,baseline_full,finetuned_full", flush=True)
    for path in sorted((args.data / "real_holdout").glob("*.jpg")):
        with Image.open(path) as source:
            image = source.convert("RGB")
        x = tf(image)[None].to(device)
        scores = []
        with torch.no_grad():
            for model in models.values():
                p = torch.softmax(model(x), dim=1)[0, 0].item()
                scores.append(round(p, 5))
        print(f"{path.name},{int(path.name in PHOTO_POSITIVE)},{scores[0]},{scores[1]}", flush=True)


if __name__ == "__main__":
    main()
