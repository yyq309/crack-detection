#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2 分割：在 CrackForest 分割数据集上训练轻量 U-Net（像素级裂缝掩膜）。
输出: outputs/best_seg.pth + outputs/seg_metrics.json + outputs/seg_samples.png
"""
import os, glob, json, random, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader
random.seed(0); np.random.seed(0); torch.manual_seed(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG  = os.path.join(ROOT, "data_seg"); OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", 40)); BATCH = int(os.environ.get("BATCH", 16))
LR = 3e-4; PATIENCE = 8; NUM_WORKERS = 0
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)

class SegDS(Dataset):
    def __init__(self, split, aug=False):
        self.f = sorted(glob.glob(f"{SEG}/{split}/img/*.png")); self.aug = aug
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        p = self.f[i]; m = p.replace("/img/","/mask/")
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        msk = (cv2.imread(m, 0) > 127).astype(np.float32)
        if self.aug:
            if random.random()<0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random()<0.5: img, msk = img[::-1], msk[::-1]
            k = random.randint(0,3)
            if k: img, msk = np.rot90(img,k), np.rot90(msk,k)
            if random.random()<0.7:
                a = random.uniform(0.8,1.2); b = random.uniform(-25,25)
                img = np.clip(img.astype(np.float32)*a+b, 0, 255).astype(np.uint8)
        x = (img.astype(np.float32)/255.0 - MEAN)/STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))), \
               torch.from_numpy(np.ascontiguousarray(msk))[None]

def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                   nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        b=base
        self.e1,self.e2,self.e3,self.e4 = blk(3,b), blk(b,b*2), blk(b*2,b*4), blk(b*4,b*8)
        self.pool = nn.MaxPool2d(2); self.bott = blk(b*8,b*16)
        self.u4=nn.ConvTranspose2d(b*16,b*8,2,2); self.d4=blk(b*16,b*8)
        self.u3=nn.ConvTranspose2d(b*8,b*4,2,2);  self.d3=blk(b*8,b*4)
        self.u2=nn.ConvTranspose2d(b*4,b*2,2,2);  self.d2=blk(b*4,b*2)
        self.u1=nn.ConvTranspose2d(b*2,b,2,2);    self.d1=blk(b*2,b)
        self.out=nn.Conv2d(b,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        z=self.bott(self.pool(e4))
        d=self.d4(torch.cat([self.u4(z),e4],1)); d=self.d3(torch.cat([self.u3(d),e3],1))
        d=self.d2(torch.cat([self.u2(d),e2],1)); d=self.d1(torch.cat([self.u1(d),e1],1))
        return self.out(d)

def dice_loss(logit, y, eps=1.0):
    p = torch.sigmoid(logit)
    num = 2*(p*y).sum((1,2,3))+eps; den = p.sum((1,2,3))+y.sum((1,2,3))+eps
    return (1-num/den).mean()

@torch.no_grad()
def evaluate(model, dl, th=0.5):
    model.eval(); I=U=TP=FP=FN=0
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        p = (torch.sigmoid(model(x))>th).float()
        I += (p*y).sum().item(); U += ((p+y)>0).float().sum().item()
        TP += (p*y).sum().item(); FP += (p*(1-y)).sum().item(); FN += ((1-p)*y).sum().item()
    iou = I/max(U,1); prec = TP/max(TP+FP,1); rec = TP/max(TP+FN,1)
    f1 = 2*prec*rec/max(prec+rec,1e-9)
    return {"IoU":iou, "Dice":f1, "P":prec, "R":rec}

def main():
    dtr = DataLoader(SegDS("train",True), BATCH, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    dva = DataLoader(SegDS("val"), BATCH, num_workers=NUM_WORKERS)
    dte = DataLoader(SegDS("test"), BATCH, num_workers=NUM_WORKERS)
    print(f"设备 {device} | train {len(dtr.dataset)} val {len(dva.dataset)} test {len(dte.dataset)}")
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    bce = nn.BCEWithLogitsLoss()
    best, bad, t0 = -1, 0, time.time()
    for ep in range(1, EPOCHS+1):
        model.train(); tot=0
        for x,y in dtr:
            x,y = x.to(device), y.to(device)
            logit = model(x); loss = bce(logit,y) + dice_loss(logit,y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sch.step()
        m = evaluate(model, dva)
        print(f"ep{ep:02d} loss {tot/len(dtr):.4f} | val IoU {m['IoU']:.4f} Dice {m['Dice']:.4f}")
        if m["IoU"] > best:
            best, bad = m["IoU"], 0
            torch.save({"state_dict":model.state_dict(),"mean":MEAN.tolist(),"std":STD.tolist(),
                        "patch":192,"val_iou":best}, f"{OUT}/best_seg.pth")
        else:
            bad += 1
            if bad >= PATIENCE: print(f"早停于 ep{ep}"); break
    model.load_state_dict(torch.load(f"{OUT}/best_seg.pth", map_location=device)["state_dict"])
    te = evaluate(model, dte)
    print(f"\n[测试集] IoU {te['IoU']:.4f} | Dice {te['Dice']:.4f} | P {te['P']:.4f} | R {te['R']:.4f}"
          f" | 用时 {(time.time()-t0)/60:.1f} 分钟")
    json.dump({"val_best_IoU":best,"test":te}, open(f"{OUT}/seg_metrics.json","w"), indent=1)

    # 定性样例
    ds = SegDS("test"); idx = list(range(0,len(ds), max(1,len(ds)//6)))[:6]
    rows=[]
    model.eval()
    for i in idx:
        x,y = ds[i]
        with torch.no_grad(): p = torch.sigmoid(model(x[None].to(device)))[0,0].cpu().numpy()
        img = np.clip(((x.numpy().transpose(1,2,0)*STD)+MEAN)*255,0,255).astype(np.uint8)
        gt  = cv2.cvtColor((y[0].numpy()*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        pr  = cv2.cvtColor(((p>0.5)*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        rows.append(np.hstack([img, gt, pr]))
    cv2.imwrite(f"{OUT}/seg_samples.png", cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(f"定性样例(原图|真值|预测) -> {OUT}/seg_samples.png")

if __name__ == "__main__":
    main()
