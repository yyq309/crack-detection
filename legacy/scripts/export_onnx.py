#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出训练好的模型为 ONNX（便于边缘/跨框架部署）。产出: outputs/model.onnx"""
import os, torch, torch.nn as nn
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
ck = torch.load(os.path.join(OUT,"best_model.pth"), map_location="cpu", weights_only=False)
IMG = ck["img_size"]; classes = ck["classes"]
model = models.resnet18()
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
model.load_state_dict(ck["state_dict"]); model.eval()

dummy = torch.randn(1,3,IMG,IMG)
path = os.path.join(OUT,"model.onnx")
# 清理旧的分离权重文件
for f in ["model.onnx","model.onnx.data"]:
    fp=os.path.join(OUT,f)
    if os.path.exists(fp): os.remove(fp)
prog = torch.onnx.export(model, (dummy,), input_names=["input"], output_names=["logits"],
                         dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}}, dynamo=True)
prog.save(path, external_data=False)          # 单文件, 权重内联
print(f"✓ 导出 ONNX -> {path}  ({os.path.getsize(path)//1024//1024}MB, 单文件)")
print(f"  输入 1x3x{IMG}x{IMG}  输出 logits[{len(classes)}]  类别={classes}")
