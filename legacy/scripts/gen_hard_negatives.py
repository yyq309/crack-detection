#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""硬负样本生成：结构接缝（板块拼接/铺装缝/砖缝/网格缝）→ 教模型"直线规则缝≠裂缝"。

两条来源（都不使用 23 张评测照片，避免泄漏）：
  A. 合成：在 CrackForest 真实 nocrack 背景上，程序化画【直线/平行/网格/砖缝】接缝；
  B. 真实：从"接缝负样本训练/"里下载的新接缝图切块（与评测集不同的图）。
接缝的判别特征=直、规则、等宽、常成网格/平行——与裂缝的蜿蜒/分叉/变宽相反。

产出：
  分类器： data/{train,val,test}/nocrack/joint_*.jpg          (128px)
  分割：   data_seg/{train,val,test}/{img,mask}/joint_*.png    (192px，掩膜全 0)
"""
import os, glob, random, math
import numpy as np, cv2
random.seed(123); np.random.seed(123)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_JOINTS = os.path.join(ROOT, "真实照片测试", "接缝负样本训练")
# 每个 split 目标合成数量（分类器）
N_CLS = {"train":900, "val":180, "test":180}
N_SEG = {"train":420, "val":90,  "test":90}

def bg_patches(size):
    """收集 CrackForest nocrack 背景块（真实混凝土/沥青纹理）。"""
    pool=[]
    for sp in ["train","val","test"]:
        for p in glob.glob(f"{ROOT}/data/{sp}/nocrack/*.jpg"):
            if os.path.basename(p).startswith("joint_"): continue
            pool.append(p)
    random.shuffle(pool); return pool

def draw_line(img, p1, p2, thick, dark, soft):
    """在 img 上画一条深色接缝：软边 + 轻噪声，模拟填缝阴影。"""
    h,w = img.shape[:2]
    m = np.zeros((h,w), np.float32)
    cv2.line(m, p1, p2, 1.0, thickness=thick, lineType=cv2.LINE_AA)
    if soft>0: m = cv2.GaussianBlur(m, (0,0), soft)
    m *= (0.85 + 0.3*np.random.rand(h,w).astype(np.float32))   # 沿缝深浅不均
    out = img.astype(np.float32)
    out -= (m[...,None] * dark)                                # 变暗
    # 偶尔加一侧高光（铺装块倒角）
    if random.random()<0.35:
        hi = np.roll(m, random.choice([-2,-1,1,2]), axis=random.choice([0,1]))
        out += (np.clip(hi-m,0,1)[...,None] * (dark*0.4))
    return np.clip(out,0,255).astype(np.uint8)

def synth_joint(bg):
    """在背景上画一种接缝形态，返回图（掩膜恒为0，因为接缝不是裂缝）。"""
    h,w = bg.shape[:2]; img = bg.copy()
    # 只保留【周期性】接缝(平行/网格/砖缝)——单条直线与真实直裂缝混淆,已剔除,
    # 否则模型会学成"见直暗线就不报",连直裂缝一起压掉(实测会让召回崩)。
    kind = random.choices(["parallel","grid","brick"], weights=[3,4,4])[0]
    dark = random.uniform(35, 95); thick = random.randint(2,6); soft = random.uniform(0.4,1.4)
    def full_line(angle_deg, offset):
        a = math.radians(angle_deg); dx,dy = math.cos(a), math.sin(a)
        cx,cy = w/2 + offset*(-dy), h/2 + offset*dx
        L=max(h,w)
        return (int(cx-dx*L),int(cy-dy*L)), (int(cx+dx*L),int(cy+dy*L))
    if kind=="single":
        ang = random.uniform(0,180)
        p1,p2 = full_line(ang, random.uniform(-w*0.2,w*0.2))
        img = draw_line(img,p1,p2,thick,dark,soft)
    elif kind=="parallel":
        ang = random.uniform(0,180); base=random.uniform(-w*0.3,w*0.3); gap=random.randint(28,70)
        for k in range(random.randint(2,3)):
            p1,p2=full_line(ang, base+k*gap); img=draw_line(img,p1,p2,thick,dark,soft)
    elif kind=="grid":
        ang=random.uniform(-15,15)
        for off in range(-w, w, random.randint(45,85)):
            p1,p2=full_line(ang, off); img=draw_line(img,p1,p2,thick,dark,soft)
        for off in range(-h, h, random.randint(45,85)):
            p1,p2=full_line(ang+90, off); img=draw_line(img,p1,p2,thick,dark,soft)
    elif kind=="brick":
        rh=random.randint(30,50)
        for r,y in enumerate(range(0,h,rh)):
            img=draw_line(img,(0,y),(w,y),thick,dark,soft)              # 横缝
            off = (r%2)*random.randint(20,40); bw=random.randint(50,80)
            for x in range(off, w, bw):
                img=draw_line(img,(x,y),(x,y+rh),thick,dark,soft)       # 竖缝
    else:  # tjunc / 角
        ang=random.uniform(0,180)
        p1,p2=full_line(ang,random.uniform(-20,20)); img=draw_line(img,p1,p2,thick,dark,soft)
        p1,p2=full_line(ang+90,random.uniform(-20,20)); img=draw_line(img,p1,p2,thick,dark,soft)
    if random.random()<0.5: img=cv2.GaussianBlur(img,(3,3),0)           # 整体轻软化
    return img

def real_joint_patches(size, per_img=30):
    """从新下载的真实接缝图切块。"""
    out=[]
    imgs=sorted(glob.glob(f"{REAL_JOINTS}/*.jpg"))
    for p in imgs:
        im=cv2.imread(p)
        if im is None: continue
        H,W=im.shape[:2]
        if min(H,W)<size: im=cv2.resize(im,(max(size,W),max(size,H))); H,W=im.shape[:2]
        for _ in range(per_img):
            y=random.randint(0,H-size); x=random.randint(0,W-size)
            out.append(im[y:y+size, x:x+size].copy())
    random.shuffle(out); return out

def main():
    pool128 = bg_patches(128); pool192 = bg_patches(192)
    real128 = real_joint_patches(128); real192 = real_joint_patches(192)
    print(f"背景块 {len(pool128)} | 真实接缝切块 128:{len(real128)} 192:{len(real192)}")

    # ---- 分类器负样本 (128) ----
    ri=0
    for sp,n in N_CLS.items():
        outdir=f"{ROOT}/data/{sp}/nocrack"; os.makedirs(outdir,exist_ok=True)
        nreal=int(n*0.35)
        for i in range(n):
            if i<nreal and ri<len(real128):
                patch=real128[ri]; ri+=1; tag="real"
            else:
                bg=cv2.imread(random.choice(pool128));
                if bg is None: continue
                bg=cv2.resize(bg,(128,128)); patch=synth_joint(bg); tag="syn"
            cv2.imwrite(f"{outdir}/joint_{tag}_{sp}_{i:04d}.jpg", patch)
        print(f"  分类器 {sp}: +{n} 接缝负样本 (真实{min(nreal,ri)}+合成)")

    # ---- 分割负样本 (192, 掩膜全0) ----
    ri=0
    for sp,n in N_SEG.items():
        di=f"{ROOT}/data_seg/{sp}/img"; dm=f"{ROOT}/data_seg/{sp}/mask"
        os.makedirs(di,exist_ok=True); os.makedirs(dm,exist_ok=True)
        zero=np.zeros((192,192),np.uint8); nreal=int(n*0.35)
        for i in range(n):
            if i<nreal and ri<len(real192):
                patch=real192[ri]; ri+=1; tag="real"
            else:
                bg=cv2.imread(random.choice(pool192))
                if bg is None: continue
                bg=cv2.resize(bg,(192,192)); patch=synth_joint(bg); tag="syn"
            cv2.imwrite(f"{di}/joint_{tag}_{sp}_{i:04d}.png", patch)
            cv2.imwrite(f"{dm}/joint_{tag}_{sp}_{i:04d}.png", zero)
        print(f"  分割   {sp}: +{n} 接缝负样本 (掩膜全0)")
    print("完成。")

if __name__=="__main__":
    main()
