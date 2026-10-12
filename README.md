# Crack Detection: 二分类识别与裂缝等级评估

本仓库整理自范智超的裂缝检测交接项目，并加入石缝负场景实验与可直接运行的串联入口。完整流程为：

```text
照片 → ResNet18 二分类（裂缝 / 非裂缝）
                    ├─ 非裂缝：停止，不进行等级评估
                    └─ 裂缝：U-Net 分割 → 宽度、长度、形态量化 → LV0–LV3 分级
```

石缝增强候选权重是**实验模型**，默认仍使用原二分类权重。原分割模型和分级规则保持不变。新增 [DamCrack 坝体裂缝样本整理与实验](experiments/README.md) 和 [Gemini 335 现场采集建议](docs/gemini335_field_capture.md)；公开坝体图片仅进入离线训练数据，不包含深度或 LV 等级标注。

## 快速运行

需要 Python 3.10+。安装与 PyTorch 兼容的运行环境后，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python scripts/detect_and_grade.py path/to/photo.jpg
```

多张照片可连续传入。结果保存在 `outputs/results.json`，判为裂缝的照片另有 `*_mask_overlay.png` 分割叠加图。只运行二分类：

```bash
python scripts/predict_binary.py path/to/photo.jpg
```

比较石缝负场景实验权重：

```bash
python scripts/detect_and_grade.py --classifier models/classifier_stone_joints_experimental.pth path/to/photo.jpg
```

`results.json` 中的 `crack_detected` 是二分类结论。`grade_status=no_crack` 表示已在第一步停止；`grade_status=graded` 表示已完成分割与分级；`grade_status=ungradable_no_mask` 表示二分类阳性、但分割未得到可测区域，此时不会强行给出等级。等级 0 可以表示已检出的发丝级裂缝，需结合 `crack_detected` 理解。

可以通过 `--class-threshold`、`--min-hits` 和 `--seg-threshold` 控制推理阈值。默认值分别为 0.6、2 个命中窗口和 0.5，与交接实验的设置一致。

### 等级与标定

`grade.py` 使用分割掩膜的 95 分位裂缝宽度，并结合范围与形态给出 0–3 级及处置描述。**未提供相机标定时，等级使用工作图像上的像素相对阈值**，不能作为实际毫米宽度或正式工程安全结论。若已知缩放到工作图像后对应的毫米/像素比，可传入 `--mm-per-px`；默认工作图像长边为 480 像素。`--dam-mode` 需要额外确认坝轴方向，可配合 `--dam-axis-deg` 使用。

## 模型与验证

| 权重 | 作用 | 状态 |
|---|---|---|
| `models/classifier_original.pth` | ResNet18 二分类 | 默认 |
| `models/segmenter_original.pth` | U-Net 裂缝分割 | 默认 |
| `models/classifier_stone_joints_experimental.pth` | 石砌体/砂浆缝负样本微调 | 实验候选，不默认启用 |

原交接包记录的分类 patch 测试准确率为 99.03%（1,134 张），分割测试 IoU 为 0.555、Dice 为 0.714；见 `models/*_archived_metrics.json`。跨场景效果低于同源 patch 测试。对交接包的 23 张现场照片，原二分类模型检出 12/14 张人工标注裂缝，同时误报 3/9 张负照片；实验候选检出 13/14，误报仍为 3/9。现场样本量很小，等级预测也依赖二分类和分割两步的正确性。详见 [实验记录](experiments/README.md)。

整条串联流程已在本地对 23 张现场照片运行。候选分类器判阳性的 16 张照片，其 LV 等级均与交接包原有 `auto_grade_real.json` 相同；其余照片不进入等级评估。测试照片和生成图因来源、隐私及体积考虑未上传。

## 仓库结构

- `crack_detection/`：二分类扫描、U-Net 分割、量化与分级流程。
- `scripts/`：可直接运行的推理入口。
- `models/`：两个原模型和一个明确标注的实验候选；见 [模型说明](models/README.md)。
- `experiments/`：石缝和坝体裂缝实验代码、指标与数据来源说明。公开数据需从原始来源取得。
- `legacy/scripts/`：交接包原始研究脚本，保留训练历史；其中目录假设依赖原交接包，不作为新仓库的运行入口。

原交接包未附可确认的统一开源许可证。这里保留其作者归属，也没有替原始代码、模型或第三方数据声明新的许可。对外再分发或商用前应确认相应权利。
