#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单图推理 demo：输入一张图 -> 输出 有裂缝/无裂缝 + 置信度 + 耗时。
用法: python scripts/predict.py <图片路径> [更多图片...]
"""
import os, sys, time, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms, models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "outputs", "best_model.pth")
device = "mps" if torch.backends.mps.is_available() else "cpu"

if not os.path.exists(CKPT):
    print("!! 未找到模型，请先训练：python scripts/train.py"); sys.exit(1)
ck = torch.load(CKPT, map_location=device, weights_only=False)
classes = ck["classes"]; IMG = ck["img_size"]
tf = transforms.Compose([transforms.Resize((IMG,IMG)), transforms.ToTensor(),
                         transforms.Normalize(ck["mean"], ck["std"])])
model = models.resnet18()
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
model.load_state_dict(ck["state_dict"]); model.to(device).eval()
CN = {"crack":"有裂缝", "nocrack":"无裂缝"}

paths = sys.argv[1:]
if not paths:
    print("用法: python scripts/predict.py <图片路径> [...]"); sys.exit(0)
for p in paths:
    if not os.path.exists(p): print(f"{p}: 文件不存在"); continue
    img = Image.open(p).convert("RGB"); x = tf(img).unsqueeze(0).to(device)
    t0=time.time()
    with torch.no_grad(): prob = torch.softmax(model(x),1)[0].cpu()
    dt=(time.time()-t0)*1000
    idx = int(prob.argmax()); name = classes[idx]
    print(f"{os.path.basename(p):28s} -> {CN.get(name,name)}  置信度 {prob[idx]*100:.1f}%   ({dt:.0f}ms)")
