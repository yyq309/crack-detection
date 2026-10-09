#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实照片裂缝检测（滑窗版）。

为什么不用 predict.py？predict.py 把整张图直接缩到 128px 再分类——手机拍的大图
(3000px+) 里一条细裂缝缩到 128px 就糊没了，会误判"无裂缝"。本脚本改成滑窗扫描：
把照片按 128px 窗口逐块分类（和模型训练方式一致），再综合出整图结论，并画出裂缝位置。

用法:
    python scripts/predict_photo.py 照片1.jpg [照片2.jpg ...]
    # 可调: LONG=1024 STRIDE=64 TH=0.6 MINHITS=2 python scripts/predict_photo.py 照片.jpg

输出:
    终端: 每张图 -> 有裂缝/无裂缝 + 裂缝窗口数/占比 + 峰值置信度 + 耗时
    文件: 同目录下 <原名>_检测结果.jpg （原图叠加裂缝热区 + 中文结论横幅）
"""
import os, sys, time, glob
import numpy as np
import torch, torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "outputs", "best_model.pth")
device = "mps" if torch.backends.mps.is_available() else "cpu"

# —— 可调参数（环境变量）——
# 多尺度扫描：同一张图在几个缩放长边下分别滑窗，任一尺度证据够即判有裂缝。
# 这样细裂缝(粗糙沥青上)和粗裂缝都不会因单一缩放尺度而漏检。
SCALES  = [int(s) for s in os.environ.get("SCALES", "512,768,1024").split(",")]
STRIDE  = int(os.environ.get("STRIDE", 64))     # 滑窗步长
TH      = float(os.environ.get("TH", 0.6))      # 判定"该窗口有裂缝"的置信度阈值
MINHITS = int(os.environ.get("MINHITS", 2))     # 整图判"有裂缝"所需的裂缝窗口数下限

FONT_PATH = "/Users/sky/Library/Fonts/NotoSansCJKsc-Bold.otf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/System/Library/Fonts/PingFang.ttc"

if not os.path.exists(CKPT):
    print("!! 未找到模型，请先训练：python scripts/train.py"); sys.exit(1)

ck = torch.load(CKPT, map_location=device, weights_only=False)
classes = ck["classes"]; IMG = ck["img_size"]
CRACK_IDX = classes.index("crack") if "crack" in classes else 0
norm = transforms.Normalize(ck["mean"], ck["std"])
model = models.resnet18()
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
model.load_state_dict(ck["state_dict"]); model.to(device).eval()


def load_font(sz):
    try:    return ImageFont.truetype(FONT_PATH, sz)
    except Exception: return ImageFont.load_default()


def scan(pil_img, long_side):
    """在指定长边下滑窗扫描，返回 (裂缝概率数组, 窗口左上角坐标, 缩放后的PIL图)。"""
    w0, h0 = pil_img.size
    scale = long_side / max(w0, h0) if max(w0, h0) > long_side else 1.0
    W, H = int(round(w0 * scale)), int(round(h0 * scale))
    img = pil_img.resize((W, H))
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC

    xs = list(range(0, max(1, W - IMG + 1), STRIDE)) or [0]
    ys = list(range(0, max(1, H - IMG + 1), STRIDE)) or [0]
    if xs[-1] != W - IMG and W >= IMG: xs.append(W - IMG)
    if ys[-1] != H - IMG and H >= IMG: ys.append(H - IMG)

    batch, coords = [], []
    for y in ys:
        for x in xs:
            patch = arr[y:y+IMG, x:x+IMG]
            if patch.shape[0] != IMG or patch.shape[1] != IMG:
                patch = np.asarray(Image.fromarray((patch*255).astype(np.uint8)).resize((IMG, IMG))).astype(np.float32)/255.0
            t = torch.from_numpy(patch).permute(2, 0, 1)
            batch.append(norm(t)); coords.append((x, y))

    probs = []
    with torch.no_grad():
        for i in range(0, len(batch), 128):
            xb = torch.stack(batch[i:i+128]).to(device)
            p = torch.softmax(model(xb), 1)[:, CRACK_IDX].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs), coords, img, (len(ys), len(xs))


def annotate(img, probs, coords, verdict, ratio, peak):
    """在缩放图上叠加裂缝热区（红框）+ 顶部中文结论横幅。"""
    W, H = img.size
    over = img.convert("RGB").copy()
    dr = ImageDraw.Draw(over, "RGBA")
    for (x, y), p in zip(coords, probs):
        if p >= TH:
            a = int(60 + 120 * min(1.0, (p - TH) / (1 - TH + 1e-6)))
            dr.rectangle([x, y, x+IMG, y+IMG], fill=(255, 40, 40, a))
    # 峰值窗口描边
    if len(probs):
        bx, by = coords[int(np.argmax(probs))]
        dr.rectangle([bx, by, bx+IMG, by+IMG], outline=(255, 215, 0, 255), width=4)

    banner_h = max(46, H // 12)
    canvas = Image.new("RGB", (W, H + banner_h), (24, 24, 28))
    canvas.paste(over, (0, banner_h))
    dr2 = ImageDraw.Draw(canvas)
    color = (255, 90, 90) if verdict == "有裂缝" else (90, 210, 130)
    f = load_font(int(banner_h * 0.5))
    fs = load_font(int(banner_h * 0.34))
    dr2.text((14, banner_h*0.12), f"检测结论：{verdict}", font=f, fill=color)
    txt = f"裂缝区占比 {ratio*100:.1f}%  峰值置信度 {peak*100:.1f}%"
    tw = dr2.textlength(txt, font=fs)
    dr2.text((W - tw - 14, banner_h*0.3), txt, font=fs, fill=(220, 220, 220))
    return canvas


def main():
    paths = []
    for a in sys.argv[1:]:
        paths += sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
    if not paths:
        print("用法: python scripts/predict_photo.py <照片路径> [...]"); return

    print(f"设备 {device} | 多尺度长边 {SCALES}px | 窗口 {IMG}px 步长 {STRIDE} | 阈值 {TH} | 判定下限 {MINHITS} 窗口\n")
    for p in paths:
        if not os.path.exists(p):
            print(f"{p}: 文件不存在"); continue
        img0 = Image.open(p).convert("RGB")
        t0 = time.time()
        # 逐尺度扫描，取"证据最强"的尺度(裂缝窗口最多)作为结论与可视化依据
        best = None
        for L in SCALES:
            probs, coords, simg, _ = scan(img0, L)
            hits = int((probs >= TH).sum())
            peak = float(probs.max()) if len(probs) else 0.0
            cand = (hits, peak, probs, coords, simg, len(probs), L)
            if best is None or hits > best[0] or (hits == best[0] and peak > best[1]):
                best = cand
        dt = (time.time() - t0) * 1000
        hits, peak, probs, coords, simg, ntot, L = best
        ratio = hits / max(1, ntot)
        verdict = "有裂缝" if hits >= MINHITS else "无裂缝"

        out = os.path.splitext(p)[0] + "_检测结果.jpg"
        annotate(simg, probs, coords, verdict, ratio, peak).save(out, quality=90)
        mark = "🔴" if verdict == "有裂缝" else "🟢"
        print(f"{mark} {os.path.basename(p):26s} -> {verdict}  |  裂缝窗口 {hits}/{ntot} "
              f"(占比 {ratio*100:.1f}%)  峰值 {peak*100:.1f}%  |  命中尺度 {L}px  |  {dt:.0f}ms")
        print(f"     标注图已保存: {out}")

    print("\n⚠ 说明：本模型训练域=CrackForest 路面。若拍的是墙面/瓷砖等差异较大的表面，"
          "结论仅供参考；正式落地需加堤坝/多域数据微调。建议先拍清晰的路面/混凝土裂缝做验证。")


if __name__ == "__main__":
    main()
