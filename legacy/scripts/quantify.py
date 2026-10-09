#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T3 量化：由裂缝二值掩膜计算 宽度/长度/面积/形态类型 等定量指标。

宽度用中轴变换(medial axis)：骨架点处的距离变换值 ×2 即为该处裂缝宽度(像素)，
这是文献中标准做法之一(等价于最大内切圆直径)。
若提供 mm_per_px(相机标定)，同时输出毫米值；未标定则只给像素值并标记 calibrated=False。
"""
import numpy as np
from skimage.morphology import medial_axis
from skimage.measure import label, regionprops
from scipy.ndimage import convolve

MIN_OBJ_PX = 30          # 小于该面积的连通域视为噪声去除

def clean_mask(mask, min_obj=MIN_OBJ_PX):
    """去除小于 min_obj 像素的连通域（自行实现，避免 skimage API 变更）。"""
    m = mask.astype(bool)
    if m.sum() == 0: return m
    lab = label(m)
    counts = np.bincount(lab.ravel())
    keep = np.zeros(counts.size, bool)
    keep[1:] = counts[1:] >= min_obj
    return keep[lab]

def quantify(mask, mm_per_px=None):
    """输入 bool/0-1 掩膜 -> 定量指标 dict。"""
    m = clean_mask(mask)
    H, W = m.shape
    out = {"calibrated": mm_per_px is not None, "mm_per_px": mm_per_px,
           "img_px": int(H*W), "area_px": 0, "area_ratio": 0.0,
           "max_width_px": 0.0, "p95_width_px": 0.0, "mean_width_px": 0.0,
           "length_px": 0.0, "n_components": 0, "n_branch": 0,
           "branch_density": 0.0, "elongation": 0.0, "orientation_deg": None,
           "crack_type": "无裂缝"}
    if m.sum() == 0:
        return out

    skel, dist = medial_axis(m, return_distance=True)
    widths = 2.0 * dist[skel]                      # 每个骨架点处的裂缝宽度(px)
    length = float(skel.sum())                     # 骨架像素数 ≈ 裂缝总长(px)
    lab = label(m); props = regionprops(lab)
    big = max(props, key=lambda r: r.area)
    # 分叉点：骨架上邻居数≥3
    nb = convolve(skel.astype(np.uint8), np.ones((3,3), np.uint8), mode="constant") - 1
    n_branch = int((skel & (nb >= 3)).sum())

    out.update({
        "area_px": int(m.sum()), "area_ratio": float(m.sum())/(H*W),
        "max_width_px": float(widths.max()), "p95_width_px": float(np.percentile(widths,95)),
        "mean_width_px": float(widths.mean()), "length_px": length,
        "n_components": int(lab.max()), "n_branch": n_branch,
        "branch_density": n_branch/max(length,1.0),
        "elongation": float(big.axis_major_length/max(big.axis_minor_length,1e-6)),
        "largest_frac": float(big.area)/float(m.sum()),   # 最大连通域占比：判"单条"还是"成片"
        "orientation_deg": float(np.degrees(big.orientation)),
    })
    out["crack_type"] = infer_type(out)
    if mm_per_px:
        for k in ["max_width","p95_width","mean_width"]:
            out[k+"_mm"] = out[k+"_px"]*mm_per_px
        out["length_mm"] = length*mm_per_px
    return out

def infer_type(q):
    """由形态特征判裂缝类型（阈值取自 CrackForest 像素真值实测分布：
    真实路面线状裂缝的 branch_density 中位 0.041、largest_frac 中位 0.99、elongation 中位 8.97，
    故"成片网状"必须以【主连通域不占优 + 组件多 + 覆盖广】为准，不能只看分叉密度）。"""
    if q["area_px"] == 0: return "无裂缝"
    lf, el, nc = q["largest_frac"], q["elongation"], q["n_components"]
    ar, bd = q["area_ratio"], q["branch_density"]
    if lf < 0.50 and nc >= 5 and ar >= 0.03: return "网状/龟裂"
    if ar >= 0.08 and bd >= 0.05:            return "网状/龟裂"
    if lf >= 0.60 and el >= 5.0:             return "线状裂缝"
    if nc >= 4 or bd >= 0.055:               return "分叉裂缝"
    if el >= 3.0:                            return "线状裂缝"
    return "局部裂缝"

def orientation_label(deg, dam_axis_deg=0.0, tol=35.0):
    """相对坝轴给出 横向/纵向（堤坝场景用）。deg 为裂缝主轴角度。"""
    if deg is None: return None
    d = abs((deg - dam_axis_deg + 90) % 180 - 90)   # 与坝轴夹角 0..90
    if d <= tol:      return "纵向(平行坝轴)"
    if d >= 90-tol:   return "横向(垂直坝轴)"
    return "斜向"
