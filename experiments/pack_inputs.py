"""Build a portable training bundle without changing the handoff package."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=Path(__file__).parent / "source_data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "training_bundle.zip")
    args = parser.parse_args()
    root = args.handoff
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    def add_file(z: zipfile.ZipFile, source: Path, dest: str, group: str) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        z.write(source, dest)
        counts[group] = counts.get(group, 0) + 1

    def add_zip_images(z: zipfile.ZipFile, source: Path, mapping, group: str) -> None:
        with zipfile.ZipFile(source) as upstream:
            for name in upstream.namelist():
                dest = mapping(name)
                if dest:
                    z.writestr(dest, upstream.read(name), compress_type=zipfile.ZIP_STORED)
                    counts[group] = counts.get(group, 0) + 1

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as bundle:
        base = root / "04_数据集" / "分类patch"
        for split in ("train", "val", "test"):
            for cls in ("crack", "nocrack"):
                for path in sorted((base / split / cls).glob("*.jpg")):
                    add_file(bundle, path, f"base/{split}/{cls}/{path.name}", f"base_{split}_{cls}")

        joints = root / "04_数据集" / "真实照片测试" / "接缝负样本训练"
        for path in sorted(joints.glob("*.jpg")):
            if path.name == "granite_1.jpg":  # This file is a book cover, not a stone joint.
                continue
            add_file(bundle, path, f"local_joints/{path.name}", "local_joints")

        real = root / "04_数据集" / "真实照片测试" / "铁塔场景验证"
        result_names = json.loads((real / "model_results.json").read_text(encoding="utf-8"))
        if len(result_names) != 23:
            raise ValueError(f"Expected 23 held-out photos, got {len(result_names)}")
        for name in sorted(result_names):
            add_file(bundle, real / name, f"real_holdout/{name}", "real_holdout")

        add_file(bundle, root / "03_模型" / "best_model.pth", "models/baseline.pth", "baseline_model")

        def crack_ch_map(name: str) -> str | None:
            parts = name.split("/")
            if len(parts) == 4 and parts[0] == "CRACK-CH" and parts[-1].lower().endswith(".jpg"):
                site = "naillac" if parts[1] == "Naillac" else "saint_nikolaos"
                cls = "crack" if parts[2] == "Positive" else "nocrack"
                return f"crack_ch/{site}/{cls}/{parts[-1]}"
            return None

        add_zip_images(bundle, args.sources / "CRACK-CH.zip", crack_ch_map, "crack_ch")

        def brick_map(name: str) -> str | None:
            parts = name.split("/")
            if len(parts) == 3 and parts[0] == "Brickwork_cracks_dataset" and parts[-1].lower().endswith(".jpg"):
                cls = "crack" if parts[1] == "Positive" else "nocrack"
                return f"brickwork/{cls}/{parts[-1]}"
            return None

        add_zip_images(bundle, args.sources / "Brickwork_cracks_dataset.zip", brick_map, "brickwork")

        def chambord_map(name: str) -> str | None:
            parts = name.split("/")
            if len(parts) == 4 and parts[0] == "Chambord_dataset" and parts[2] == "images" and parts[-1].lower().endswith(".png"):
                return f"chambord/{parts[1].lower()}/{parts[-1]}"
            return None

        add_zip_images(bundle, args.sources / "Chambord_dataset.zip", chambord_map, "chambord")

        manifest = {
            "counts": counts,
            "source_sha256": {
                name: sha256(args.sources / name)
                for name in ("CRACK-CH.zip", "Brickwork_cracks_dataset.zip", "Chambord_dataset.zip")
            },
            "baseline_sha256": sha256(root / "03_模型" / "best_model.pth"),
            "notes": [
                "The 23 real photos are held out from training and threshold selection.",
                "granite_1.jpg is excluded because it depicts a book cover.",
                "ncpav_1.jpg has conflicting source intent and manual grading; report both interpretations.",
                "CRACK-CH saint_nikolaos is held out by site.",
            ],
        }
        bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size, **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
