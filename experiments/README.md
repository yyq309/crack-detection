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

## Reproduction

`pack_inputs.py` expects the original handoff directory and the public dataset archives. It creates an offline training bundle with train/validation/test splits. Place that extracted bundle in `data/` and use a PyTorch environment with CUDA for training. For example:

```bash
python experiments/run_experiment.py --data data --output outputs/public_joint --epochs 8 --batch 64
```

The selected public-joint checkpoint is `models/classifier_stone_joints_experimental.pth`. `run_balanced.py` and `run_multiclass.py` are research ablations, not deployment defaults. Training data and the 23 field photos are excluded from this repository; metrics can be inspected without them.
