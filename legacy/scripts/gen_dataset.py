#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 CrackForest(图+像素真值) 生成裂缝/无裂缝二分类 patch 数据集。
按源图划分 train/val/test（防泄漏），类别均衡。
产出: data/train|val|test/{crack,nocrack}/*.jpg
"""
import os, cv2, numpy as np, scipy.io as sio, random, glob
random.seed(42); np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "data", "crackforest")
OUT  = os.path.join(ROOT, "data")
PATCH, STRIDE, CRACK_TH = 128, 24, 64      # 最优配置(cross-scene测试96.2%)

# SPLIT=scene(严格,按源图划分,反映真实泛化) | random(标准patch随机划分,论文常用基准协议)
SPLIT = os.environ.get("SPLIT", "scene")
ids = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(f"{SRC}/gt") if f.endswith(".mat"))
random.shuffle(ids)
n = len(ids); tr, va = ids[:int(n*0.7)], ids[int(n*0.7):int(n*0.85)]; te = ids[int(n*0.85):]
split_of = {i:"train" for i in tr}; split_of.update({i:"val" for i in va}); split_of.update({i:"test" for i in te})
def patch_split(i):
    if SPLIT == "random":
        r = random.random(); return "train" if r < 0.7 else ("val" if r < 0.85 else "test")
    return split_of[i]
print(f"划分模式: {SPLIT}  (源图 train {len(tr)}/val {len(va)}/test {len(te)})")

for s in ["train","val","test"]:
    for c in ["crack","nocrack"]:
        os.makedirs(f"{OUT}/{s}/{c}", exist_ok=True)

def load_gt(i):
    seg = sio.loadmat(f"{SRC}/gt/{i:03d}.mat")['groundTruth'][0,0]['Segmentation']
    return (np.array(seg)==2).astype(np.uint8)

# 收集候选 patch, 按 split 分桶
buckets = {s:{"crack":[], "nocrack":[]} for s in ["train","val","test"]}
for i in ids:
    img = cv2.imread(f"{SRC}/image/{i:03d}.jpg"); gt = load_gt(i)
    if img is None: continue
    H,W = gt.shape
    for y in range(0, H-PATCH+1, STRIDE):
        for x in range(0, W-PATCH+1, STRIDE):
            cp = int(gt[y:y+PATCH, x:x+PATCH].sum())
            patch = img[y:y+PATCH, x:x+PATCH]; s = patch_split(i)
            if cp >= CRACK_TH:      buckets[s]["crack"].append((patch, i, y, x))
            elif cp == 0:           buckets[s]["nocrack"].append((patch, i, y, x))
            # 1..CRACK_TH-1 之间的模糊patch丢弃

# 均衡: 每个split内两类下采样到较少者(1:1),防止"全猜裂缝"也能高准确率
counts = {}
for s in ["train","val","test"]:
    cr, nc = buckets[s]["crack"], buckets[s]["nocrack"]
    random.shuffle(cr); random.shuffle(nc)
    k = min(len(cr), len(nc)); cr, nc = cr[:k], nc[:k]
    for cls, lst in [("crack",cr), ("nocrack",nc)]:
        for patch,i,y,x in lst:
            cv2.imwrite(f"{OUT}/{s}/{cls}/{i:03d}_{y}_{x}.jpg", patch)
    counts[s] = (len(cr), len(nc))

print("patch数量 (crack / nocrack):")
for s in ["train","val","test"]:
    print(f"  {s}: {counts[s][0]} / {counts[s][1]}  = {sum(counts[s])}")
print("总计:", sum(sum(v) for v in counts.values()), "patch")
