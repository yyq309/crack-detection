#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2/T3 前置：从 CrackForest(图 + 像素级真值) 生成【分割】数据集(图+掩膜patch)。
按源图划分 train/val/test，防止同图泄漏。
产出: data_seg/{train,val,test}/{img,mask}/*.png
"""
import os, cv2, numpy as np, scipy.io as sio, random
random.seed(42); np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "data", "crackforest")
OUT  = os.path.join(ROOT, "data_seg")
PATCH, STRIDE = 192, 64
MIN_CRACK_PX  = 40          # 含裂缝像素数下限，才算"正样本patch"
BG_RATIO      = 0.25        # 额外保留的纯背景patch比例(相对正样本数)

ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(f"{SRC}/gt") if f.endswith(".mat"))
random.shuffle(ids)
n = len(ids)
tr, va, te = ids[:int(n*0.7)], ids[int(n*0.7):int(n*0.85)], ids[int(n*0.85):]
split_of = {**{i:"train" for i in tr}, **{i:"val" for i in va}, **{i:"test" for i in te}}
print(f"源图划分: train {len(tr)} / val {len(va)} / test {len(te)}")

for s in ["train","val","test"]:
    for k in ["img","mask"]:
        os.makedirs(f"{OUT}/{s}/{k}", exist_ok=True)

def load_gt(i):
    seg = sio.loadmat(f"{SRC}/gt/{i:03d}.mat")['groundTruth'][0,0]['Segmentation']
    return (np.array(seg)==2).astype(np.uint8)

pos = {s:[] for s in ["train","val","test"]}
bg  = {s:[] for s in ["train","val","test"]}
for i in ids:
    img = cv2.imread(f"{SRC}/image/{i:03d}.jpg")
    if img is None: continue
    gt = load_gt(i); H,W = gt.shape
    s = split_of[i]
    for y in range(0, H-PATCH+1, STRIDE):
        for x in range(0, W-PATCH+1, STRIDE):
            m = gt[y:y+PATCH, x:x+PATCH]
            rec = (i,y,x, img[y:y+PATCH, x:x+PATCH], m)
            if m.sum() >= MIN_CRACK_PX: pos[s].append(rec)
            elif m.sum() == 0:          bg[s].append(rec)

tot=0
for s in ["train","val","test"]:
    random.shuffle(bg[s])
    keep = pos[s] + bg[s][:int(len(pos[s])*BG_RATIO)]
    for i,y,x,patch,m in keep:
        cv2.imwrite(f"{OUT}/{s}/img/{i:03d}_{y}_{x}.png", patch)
        cv2.imwrite(f"{OUT}/{s}/mask/{i:03d}_{y}_{x}.png", (m*255).astype(np.uint8))
    print(f"  {s}: 正样本 {len(pos[s])} + 背景 {min(len(bg[s]),int(len(pos[s])*BG_RATIO))} = {len(keep)}")
    tot += len(keep)
print(f"分割数据集共 {tot} 个 {PATCH}x{PATCH} patch -> {OUT}")
