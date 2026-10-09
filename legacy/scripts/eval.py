#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在测试集上评测已训练模型：准确率/精确率/召回/F1 + 混淆矩阵。
产出: outputs/metrics.json, outputs/confusion_matrix.png
"""
import os, json, numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
CKPT = os.path.join(OUT, "best_model.pth")
device = "mps" if torch.backends.mps.is_available() else "cpu"

ck = torch.load(CKPT, map_location=device, weights_only=False)
classes = ck["classes"]; c2i = ck["class_to_idx"]; IMG = ck["img_size"]
crack_idx = c2i.get("crack", 0)                    # 裂缝=正类
tf = transforms.Compose([transforms.Resize((IMG,IMG)), transforms.ToTensor(),
                         transforms.Normalize(ck["mean"], ck["std"])])
test_ds = datasets.ImageFolder(os.path.join(ROOT,"data","test"), tf)
test_ld = DataLoader(test_ds, 64, shuffle=False, num_workers=0)

model = models.resnet18()
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
model.load_state_dict(ck["state_dict"]); model.to(device).eval()

TTA = os.environ.get("TTA","1") != "0"      # 测试时增强:原图+水平翻转+垂直翻转 平均
ys, ps = [], []
with torch.no_grad():
    for x,y in test_ld:
        x = x.to(device)
        logits = torch.softmax(model(x),1)
        if TTA:
            logits = logits + torch.softmax(model(torch.flip(x,[3])),1) + torch.softmax(model(torch.flip(x,[2])),1)
        ps += logits.argmax(1).cpu().tolist(); ys += y.tolist()
ys, ps = np.array(ys), np.array(ps)
print("TTA:", "on" if TTA else "off")

# 混淆矩阵 (行=真实, 列=预测), 顺序按 classes
K = len(classes); cm = np.zeros((K,K), int)
for t,p in zip(ys,ps): cm[t,p]+=1
acc = (ys==ps).mean()
# 以 crack 为正类
tp = cm[crack_idx,crack_idx]; fn = cm[crack_idx].sum()-tp
fp = cm[:,crack_idx].sum()-tp; tn = cm.sum()-tp-fn-fp
prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9); f1 = 2*prec*rec/(prec+rec+1e-9)

metrics = {"accuracy":round(float(acc),4), "precision_crack":round(float(prec),4),
           "recall_crack":round(float(rec),4), "f1_crack":round(float(f1),4),
           "n_test":int(len(ys)), "classes":classes, "confusion_matrix":cm.tolist(),
           "val_acc_at_train":ck.get("val_acc")}
json.dump(metrics, open(os.path.join(OUT,"metrics.json"),"w"), ensure_ascii=False, indent=2)

# 渲染混淆矩阵图 (cv2, 英文标签)
cell=140; pad=90; W=pad+cell*K+20; H=pad+cell*K+20
img=np.full((H,W,3),255,np.uint8)
mx=cm.max()
for r in range(K):
    for c in range(K):
        v=cm[r,c]; shade=int(255-200*v/(mx+1e-9))
        col=(shade,shade,255) if r==c else (shade,shade,shade)
        x0,y0=pad+c*cell, pad+r*cell
        cv2.rectangle(img,(x0,y0),(x0+cell,y0+cell),col,-1)
        cv2.rectangle(img,(x0,y0),(x0+cell,y0+cell),(180,180,180),1)
        cv2.putText(img,str(v),(x0+cell//2-20,y0+cell//2+8),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)
for i,cn in enumerate(classes):
    cv2.putText(img,cn[:8],(pad+i*cell+20,pad-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)   # 列(预测)
    cv2.putText(img,cn[:8],(6,pad+i*cell+cell//2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)    # 行(真实)
cv2.putText(img,"pred ->",(pad,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
cv2.putText(img,f"acc={acc:.3f}",(W-180,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,120,0),2)
cv2.imwrite(os.path.join(OUT,"confusion_matrix.png"), img)

print("="*46)
print(f"测试集 {len(ys)} 张  |  准确率 {acc:.4f}")
print(f"裂缝类  精确率 {prec:.4f}  召回率 {rec:.4f}  F1 {f1:.4f}")
print(f"混淆矩阵(行真实/列预测, 顺序{classes}):\n{cm}")
print("已保存 -> outputs/metrics.json, outputs/confusion_matrix.png")
