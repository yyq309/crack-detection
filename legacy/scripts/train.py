#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""裂缝识别二分类 · 训练脚本（ResNet18 迁移学习，M1 Pro MPS GPU）。
用法: python scripts/train.py   (所有路径/超参在下方 CONFIG 区，无需传参)
产出: outputs/best_model.pth, outputs/train.log
"""
import os, sys, time, json, datetime
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ─────────── CONFIG ───────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")           # data/train data/val data/test
OUT_DIR   = os.path.join(ROOT, "outputs"); os.makedirs(OUT_DIR, exist_ok=True)
IMG_SIZE  = 224
BATCH     = int(os.environ.get("BATCH", 64))
MAX_EPOCH = int(os.environ.get("EPOCHS", 16))
LR        = 3e-4
PATIENCE  = 5            # 验证集准确率连续不升则早停
FREEZE_EP = 1            # 前FREEZE_EP轮冻结主干只训分类头(warmup)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))   # 0=不用多进程,避开py3.14 spawn问题,小数据集无影响
SEED      = 42
SMOKE     = os.environ.get("SMOKE", "") != ""   # 冒烟测试:只跑少量batch验证流程
# ──────────────────────────────

torch.manual_seed(SEED)
LOG = open(os.path.join(OUT_DIR, "train.log"), "a")
def log(*a):
    msg = "[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), " ".join(str(x) for x in a))
    print(msg); LOG.write(msg+"\n"); LOG.flush()

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
log("="*50); log("训练启动  device =", device, " torch =", torch.__version__)

# ─────────── 数据 ───────────
IN_MEAN, IN_STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75,1.0), ratio=(0.8,1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.25,0.25,0.25),
    transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02,0.12)),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD),
])

def find_split(name_options):
    for n in name_options:
        p = os.path.join(DATA_DIR, n)
        if os.path.isdir(p): return p
    log("!! 找不到数据子目录:", name_options, " 请确认已解压。"); sys.exit(1)

train_dir = find_split(["train"])
val_dir   = find_split(["val","validation","valid"])
test_dir  = find_split(["test"])

train_ds = datasets.ImageFolder(train_dir, train_tf)
val_ds   = datasets.ImageFolder(val_dir,   eval_tf)
classes  = train_ds.classes                      # 例如 ['Negative','Positive']
log("类别:", train_ds.class_to_idx, " | train/val =", len(train_ds), "/", len(val_ds))

train_ld = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=NUM_WORKERS)
val_ld   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=NUM_WORKERS)

# ─────────── 模型 ───────────
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(classes)))
model = model.to(device)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)

def set_backbone_frozen(frozen):
    for name,p in model.named_parameters():
        if not name.startswith("fc"): p.requires_grad = not frozen
set_backbone_frozen(True)                      # warmup: 先只训分类头
opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched= torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCH)

@torch.no_grad()
def evaluate(loader):
    model.eval(); correct=total=0; loss_sum=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x); loss_sum += crit(out,y).item()*x.size(0)
        correct += (out.argmax(1)==y).sum().item(); total += x.size(0)
    return correct/total, loss_sum/total

# ─────────── 训练循环 ───────────
best_acc = 0.0; best_epoch = 0; no_improve = 0
t0 = time.time()
for ep in range(1, MAX_EPOCH+1):
    if ep == FREEZE_EP+1:
        set_backbone_frozen(False); log("   解冻主干,全网络微调")
    model.train(); run_loss=0.0; seen=0; te=time.time()
    for i,(x,y) in enumerate(train_ld):
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); out = model(x); loss = crit(out,y)
        loss.backward(); opt.step()
        run_loss += loss.item()*x.size(0); seen += x.size(0)
        if i % 20 == 0:
            log(f"  epoch {ep} step {i}/{len(train_ld)} loss={loss.item():.4f}")
        if SMOKE and i >= 3: break        # 冒烟测试:每轮只跑4个batch
    sched.step()
    va, vl = evaluate(val_ld)
    log(f"== epoch {ep}/{MAX_EPOCH}  train_loss={run_loss/seen:.4f}  val_acc={va:.4f}  val_loss={vl:.4f}  ({time.time()-te:.0f}s)")
    if va > best_acc:
        best_acc, best_epoch, no_improve = va, ep, 0
        torch.save({"state_dict":model.state_dict(), "classes":classes,
                    "class_to_idx":train_ds.class_to_idx, "img_size":IMG_SIZE,
                    "mean":IN_MEAN, "std":IN_STD, "arch":"resnet18",
                    "val_acc":va, "epoch":ep},
                   os.path.join(OUT_DIR,"best_model.pth"))
        log(f"   ✓ 保存最优模型 (val_acc={va:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            log(f"   早停：验证准确率连续 {PATIENCE} 轮未提升。"); break

log(f"训练完成。最优 val_acc={best_acc:.4f} @epoch{best_epoch}  总耗时 {time.time()-t0:.0f}s")
log("最优模型 ->", os.path.join(OUT_DIR,"best_model.pth"))
LOG.close()
