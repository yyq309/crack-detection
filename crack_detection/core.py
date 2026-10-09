"""ResNet18 binary crack classifier and multi-scale photo scanning."""

from __future__ import annotations

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def create_model(checkpoint: dict) -> nn.Module:
    classes = checkpoint.get("classes")
    if classes != ["crack", "nocrack"]:
        raise ValueError(f"Expected ['crack', 'nocrack'], got {classes}")
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def scan_photo(model: nn.Module, image: Image.Image, device: torch.device,
               *, mean: list[float], std: list[float], patch_size: int = 224,
               scales: tuple[int, ...] = (512, 768, 1024), stride: int = 64,
               threshold: float = 0.6, min_hits: int = 2) -> dict:
    """Return the strongest decision from the original multi-scale sliding scan."""
    model.eval()
    normalize = transforms.Normalize(mean, std)
    original_width, original_height = image.size
    best = None
    for long_side in scales:
        factor = min(1.0, long_side / max(original_width, original_height))
        width = max(1, round(original_width * factor))
        height = max(1, round(original_height * factor))
        resized = image.resize((width, height))
        xs = list(range(0, max(1, width - patch_size + 1), stride)) or [0]
        ys = list(range(0, max(1, height - patch_size + 1), stride)) or [0]
        if width >= patch_size and xs[-1] != width - patch_size:
            xs.append(width - patch_size)
        if height >= patch_size and ys[-1] != height - patch_size:
            ys.append(height - patch_size)
        patches = []
        for y in ys:
            for x in xs:
                patch = resized.crop((x, y, x + patch_size, y + patch_size))
                if patch.size != (patch_size, patch_size):
                    patch = patch.resize((patch_size, patch_size))
                patches.append(normalize(transforms.functional.to_tensor(patch)))
        probabilities = []
        for start in range(0, len(patches), 128):
            batch = torch.stack(patches[start:start + 128]).to(device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(batch)
            probabilities.extend(torch.softmax(logits.float(), dim=1)[:, 0].cpu().tolist())
        hits = sum(p >= threshold for p in probabilities)
        peak = max(probabilities, default=0.0)
        candidate = {"hits": hits, "peak": round(peak, 5),
                     "windows": len(probabilities), "scale": long_side,
                     "predicted_crack": hits >= min_hits}
        if best is None or (hits, peak) > (best["hits"], best["peak"]):
            best = candidate
    return best
