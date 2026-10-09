#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T4 自动分级：把 T3 量化指标映射为危险等级 0/1/2/3 + 处置动作 + 判定理由。

等级定义（与《裂缝危险分级与修补判定》一致）:
  0 无需处理 / 1 观察监测 / 2 计划修补 / 3 立即处理
宽度阈值对齐 BRE Digest 251、ACI 224R、GB 50010；
堤坝模式下额外按"横向(垂直坝轴)"加权——横向贯穿缝是渗漏/管涌主因。
"""
from .quantify import orientation_label

LEVELS = {0:("无需处理","常规巡检即可"),
          1:("观察监测","建档并定期复测，跟踪是否扩展"),
          2:("计划修补","纳入检修计划，择期灌缝、封闭或修补"),
          3:("立即处理","立即上报并安排处置")}

# 已标定(毫米)阈值：对齐规范
W1_MM, W2_MM, W3_MM = 0.2, 1.0, 5.0
# 未标定(像素)阈值：由 CrackForest 像素真值宽度分布标定（见 calib_width.py）
W1_PX, W2_PX, W3_PX = 2.0, 5.5, 8.3   # 取自CrackForest真值宽度分布:中位5.66px、P90 8.25px

def grade(q, dam_mode=False, dam_axis_deg=0.0):
    """q = quantify() 输出。返回 dict(level, name, action, reasons, width_used)。"""
    reasons = []
    if q["area_px"] == 0:
        return {"level":0, "name":LEVELS[0][0], "action":LEVELS[0][1],
                "reasons":["未检出裂缝"], "width_used":0.0, "unit":"px"}

    cal = q.get("calibrated") and q.get("mm_per_px")
    if cal:
        w = q.get("p95_width_mm", 0.0); t1,t2,t3 = W1_MM, W2_MM, W3_MM; unit="mm"
    else:
        w = q["p95_width_px"];          t1,t2,t3 = W1_PX, W2_PX, W3_PX; unit="px"

    if   w < t1: lv = 0; reasons.append(f"宽度 {w:.1f}{unit} 低于起判阈值({t1}{unit})，属发丝级")
    elif w < t2: lv = 1; reasons.append(f"宽度 {w:.1f}{unit}，细裂缝")
    elif w < t3: lv = 2; reasons.append(f"宽度 {w:.1f}{unit}，已明显开口")
    else:        lv = 3; reasons.append(f"宽度 {w:.1f}{unit}，宽/深裂缝")

    # 形态与范围修正
    if q["crack_type"] == "网状/龟裂" and q["area_ratio"] >= 0.06:
        lv += 1; reasons.append(f"网状/龟裂且面积占比 {q['area_ratio']*100:.1f}%，成片劣化")
    if q["area_ratio"] >= 0.15:
        lv += 1; reasons.append(f"裂缝面积占比 {q['area_ratio']*100:.1f}%，范围很大")
    # 注：分叉点数不作为独立加权项——实测真实线状路面裂缝分叉密度中位已达 0.041，
    # 若按分叉计数加权会对正常裂缝普遍抬级（已验证会造成等级虚高）。

    # 堤坝模式：横向(垂直坝轴)缝加权
    if dam_mode:
        ol = orientation_label(q.get("orientation_deg"), dam_axis_deg)
        if ol == "横向(垂直坝轴)":
            lv += 1; reasons.append("走向为横向(垂直坝轴)，存在贯穿渗漏/管涌风险")
        elif ol: reasons.append(f"走向：{ol}")

    lv = max(0, min(3, lv))
    return {"level":lv, "name":LEVELS[lv][0], "action":LEVELS[lv][1],
            "reasons":reasons, "width_used":round(w,2), "unit":unit}
