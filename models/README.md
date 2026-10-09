# Model checkpoints

These checkpoints come from the user's original crack-detection handoff and the subsequent stone-joint negative-scene experiment. They are PyTorch files and should be loaded only from this trusted repository.

| File | Purpose | SHA-256 |
|---|---|---|
| `classifier_original.pth` | Original ResNet18 binary crack/no-crack classifier (224 px) | `f4f1d77b935aa45b2a245ee6dae1b6ac5ed3f1f4c23a8ca619003a605babb6ae` |
| `segmenter_original.pth` | Original U-Net crack segmenter (192 px training patches) | `43690d8c3a8248ef194cbb00d444541e33e24a800c566f98eca31caf38e16979` |
| `classifier_stone_joints_experimental.pth` | Binary candidate fine-tuned with public masonry/joint negatives; research only | `19a984eb3f9c03f232b17b872bdc6d78a6ff54329432da19ca765d19b7c9c721` |

The experimental classifier reduced false positives on the public Chambord mortar-joint set from 34/92 to 0/92 but did **not** reduce false positives on the 9 negative field photos (3/9 for both classifiers). It is not the default model. See `experiments/README.md` for the full tradeoff.

The original ONNX export was omitted because it duplicates the original classifier checkpoint and is not used by the current pipeline. `legacy/scripts/export_onnx.py` preserves the export code.
