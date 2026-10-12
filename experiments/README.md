# Stone-joint negative-scene experiments

The maintained pipeline uses the original binary classifier by default. The checkpoint `models/classifier_stone_joints_experimental.pth` is an optional research candidate. Training and evaluation scripts are preserved here; image datasets are intentionally absent.

## Data provenance

- [CRACK-CH](https://zenodo.org/records/6516913): 224 px stone-masonry crack/no-crack patches. Naillac was used for training; Saint Nikolaos was evaluated by site. No explicit reuse license was found on the record, so the images are not redistributed here.
- [Chambord mortar-joint segmentation](https://huggingface.co/datasets/biglam/chambord-mortar-joint-segmentation): 333 training and 92 test stone/mortar-joint patches, CC BY 4.0. All patches come from one monument, so test images are correlated by site.
- [Brickwork Cracks](https://zenodo.org/records/8014150): 350 crack and 350 no-crack patches. Matching stems were kept in the same split. No explicit reuse license was found on the record; images are not redistributed.
- [UK masonry cracks and masks](https://zenodo.org/records/18458458): 100 original photos, CC BY 4.0. `prepare_uk_masonry.py` created positive crops, splitting 80/20 by original photo (956 train crops, 160 test crops).
- [PaveSeg GitHub preview](https://github.com/FuturePave/PaveSeg-Dataset): inspected but not used. Only 3 of the 100 preview images had expansion-joint labels; these are road joints and poorly match natural stone seams.

Eight local joint-only photos from the handoff were included in training; one mislabeled book-cover image was excluded. The original 23 field photos were not used for model training. Those photos were viewed repeatedly during model development, so they are not a pristine final test set.

## Results

| Classifier | Chambord joint false positives | Stone site: crack TP/7, no-crack TN/14 | Field photos: crack TP/14, no-crack TN/9 |
|---|---:|---:|---:|
| Original | 34/92 | 1/7, 13/14 | 12/14, 6/9 |
| Public-joint candidate | 0/92 | 2/7, 14/14 | 13/14, 6/9 |
| Mined local negatives | 0/92 | 0/7, 14/14 | 5/14, 9/9 |
| Three-class experiment | 0/92 | 0/7, 14/14 | 3/14, 9/9 |
| With UK masonry positives | 0/92 | 0/7, 14/14 | 11/14, 7/9 |

Detailed metrics are in `metrics/`. `ncpav_1.jpg` has conflicting source labels; this table uses the handoff's manual grading, which marked it as a crack. JSON files include the alternative manifest interpretation. All results are small-sample exploratory results, not a field performance guarantee.

The over-suppression experiments show why zero false positives alone is not a valid selection criterion: several true cracks disappear. Further training needs independently labeled target-domain stone joints and fine cracks, grouped by original photo and location.

## Concrete dam augmentation

[DamCrack](https://zenodo.org/records/17274707) provides 4,000 annotated 640×640 images of an actual concrete dam, under CC BY 4.0. Its segmentation archive includes crack and spalling masks. `prepare_damcrack.py` verifies the archive MD5, reads the **refined** masks, and writes only crack-positive 224×224 training crops with source-image IDs and crop coordinates in a manifest. Red mask pixels represent cracks and blue pixels represent spalling, as checked against sample images and annotations. Spalling-only and annotation-empty images are inventoried but are **not automatically treated as negative samples**. They need human review before being used for the strict “crack versus no crack” task.

The current curation run found 3,551 crack-annotated images (including 2,250 with spalling), 277 spalling-only images, and 172 annotation-empty images. It produced 7,064 positive crops from 3,549 source images with the default 100-pixel minimum. Two source images had too little annotated crack area to meet that threshold.

```bash
python experiments/prepare_damcrack.py \
  --archive /path/to/'Damage Segmentaion.zip' \
  --output /path/to/damcrack_crops
```

The released 640 px patches lack reliable original capture/site group identifiers. Keep DamCrack crops training-only in this experiment; do not claim an independent held-out dam accuracy from a random patch split. Public images are RGB, so they cannot train or validate a Gemini 335 depth branch or establish field performance for this camera. Neither source images nor derived crops are committed to this repository.

To include the curated crops in the classifier experiment, copy `damcrack_crops` to `data/damcrack` beside the existing training bundle and run `run_experiment.py`. When that directory is present, 10% of sampled training examples come from the dam crack crops. The default classifier checkpoint and grading pipeline are unchanged until a candidate passes the existing external and field checks.

Two 4090 experiments used the 7,064 DamCrack crops. The first started from the original checkpoint and included mined local joint negatives; the second started from the public-joint candidate with `--exclude-mined`, 3 epochs and a 5e-6 learning rate. The selected weights were evaluated with the existing photo scanner:

| Model | Field crack TP/14 | Field no-crack TN/9 | Saint Nikolaos crack TP/7 | Chambord joint FP/92 |
|---|---:|---:|---:|---:|
| Original | 12 | 6 | 1 | 34 |
| Public-joint candidate | 13 | 6 | 2 | 0 |
| DamCrack + mined joints | 10 | 8 | 0 | 0 |
| DamCrack conservative | 13 | 6 | 2 | 0 |

Metrics for both DamCrack experiments are in `metrics/`. The conservative run did not improve these checks over the public-joint candidate. The aggressive run lost too many true cracks. Neither is promoted as the default classifier. The 23 field photos have been consulted during development and are only exploratory checks. A separate Gemini 335 dam-site test set is still needed, and DamCrack has no LV0–LV3 grade labels for validating severity assessment.

For Gemini 335 field collection, preserve native RGB, raw 16-bit depth, depth scale, RGB-aligned depth, camera intrinsics/extrinsics, and synchronized timestamps. Record location, standoff distance, view angle, illumination/wetness, and an in-frame scale reference. Label real cracks separately from natural stone seams/construction joints, spalling, and uncertain areas; split evaluation by site and capture session before cropping. The current LV0–LV3 grade is based on RGB segmentation geometry and handoff rules, not a calibrated physical width measurement from depth.

## Reproduction

`pack_inputs.py` expects the original handoff directory and the public dataset archives. It creates an offline training bundle with train/validation/test splits. Place that extracted bundle in `data/` and use a PyTorch environment with CUDA for training. For example:

```bash
python experiments/run_experiment.py --data data --output outputs/public_joint --epochs 8 --batch 64
```

The selected public-joint checkpoint is `models/classifier_stone_joints_experimental.pth`. `run_balanced.py` and `run_multiclass.py` are research ablations, not deployment defaults. Training data and the 23 field photos are excluded from this repository; metrics can be inspected without them.
