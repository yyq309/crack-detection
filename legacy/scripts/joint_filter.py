#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接缝几何后处理：识别【板块拼接/铺装/瓷砖/砖墙】的规则接缝并剔除其误报，保留裂缝。

关键：真实照片里接缝的掩膜常是碎的/不规则的（分割器沿缝断续触发），
但接缝在【原图】里是笔直、规则、成平行组或正交网格的。所以在【原图】上用
Canny+Hough 找"规则直线组"（≥3 条平行 或 正交网格）＝接缝，再把掩膜里落在这些
接缝线上的像素抹掉。裂缝走向随机、不成规则网格，不会被抹。

局限（已知，如实交接）：不规则砌石墙（叠石/毛石）的缝本身杂乱无规则，
几何上与裂缝网络难分，本方法不处理该类。
"""
import numpy as np, cv2

L_FRAC   = 0.20     # 图内直线长度下限 = L_FRAC × 长边
ANG_TOL  = 9.0      # 同向聚类容差
MIN_PARALLEL = 3    # 判"平行接缝组"所需的同向长直线数
OFF_MIN  = 12.0     # 平行线最小间距(px)
REM_W    = 13       # 沿接缝线抹除带宽

def _ang(x1,y1,x2,y2): return np.degrees(np.arctan2(y2-y1, x2-x1)) % 180.0
def _off(a,x1,y1):
    r=np.radians(a); return x1*np.sin(r)-y1*np.cos(r)
def _close(a,b,t):
    d=abs(a-b)%180.0; return min(d,180.0-d)<=t

def _joint_lines(gray, W, H):
    """在原图上找"规则接缝直线组"，返回被判为接缝的线段列表。"""
    edges = cv2.Canny(gray, 40, 120)
    Lmin = int(L_FRAC*max(W,H))
    hl = cv2.HoughLinesP(edges,1,np.pi/180, threshold=max(40,Lmin//3),
                         minLineLength=Lmin, maxLineGap=8)
    if hl is None or len(hl) < MIN_PARALLEL:
        return []
    segs=[{"ang":_ang(*l),"l":l} for l in hl[:,0]]
    for s in segs: s["off"]=_off(s["ang"], s["l"][0], s["l"][1])
    # 角度聚类
    used=[False]*len(segs); groups=[]
    for i in range(len(segs)):
        if used[i]: continue
        g=[i]; used[i]=True
        for j in range(i+1,len(segs)):
            if not used[j] and _close(segs[i]["ang"],segs[j]["ang"],ANG_TOL):
                g.append(j); used[j]=True
        groups.append(g)
    def distinct_offsets(g):
        offs=sorted(segs[i]["off"] for i in g); k=[offs[0]]
        for o in offs[1:]:
            if o-k[-1]>=OFF_MIN: k.append(o)
        return len(k)
    keep=set(); gangs=[np.mean([segs[i]["ang"] for i in g]) for g in groups]
    # 平行接缝组：≥MIN_PARALLEL 条不同偏移的同向线
    for g in groups:
        if distinct_offsets(g) >= MIN_PARALLEL:
            keep.update(g)
    # 正交网格：两组近正交、各≥2条不同偏移
    for a in range(len(groups)):
        for b in range(a+1,len(groups)):
            d=abs(gangs[a]-gangs[b])%180.0; d=min(d,180.0-d)
            if 60<=d<=120 and distinct_offsets(groups[a])>=2 and distinct_offsets(groups[b])>=2:
                keep.update(groups[a]); keep.update(groups[b])
    return [segs[i]["l"] for i in keep]

def reject_joints(mask, image=None, return_info=False):
    m=mask.astype(bool); H,W=m.shape
    info={"joint_lines":0,"removed_px":0}
    if m.sum()==0 or image is None:
        return (m,info) if return_info else m
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim==3 else image
    lines=_joint_lines(gray,W,H)
    info["joint_lines"]=len(lines)
    if not lines:
        return (m,info) if return_info else m
    jm=np.zeros((H,W),np.uint8)
    for x1,y1,x2,y2 in lines: cv2.line(jm,(x1,y1),(x2,y2),1,REM_W)
    before=int(m.sum()); out=m & (~jm.astype(bool))
    info["removed_px"]=before-int(out.sum())
    return (out,info) if return_info else out
