#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T3+T4 端到端自动分级：图片 -> U-Net分割 -> 量化(宽度/长度/类型) -> 危险等级0-3。

用法:
  python scripts/auto_grade.py real                 # 23张真实巡检照片
  python scripts/auto_grade.py crackforest [N]      # CrackForest数据样例(默认全部)
  python scripts/auto_grade.py file <图片> [...]    # 任意图片

尺度约定：未标定时统一把图缩放到长边 480px（与 CrackForest 同尺度），
使像素宽度阈值可比；若提供 MM_PER_PX 环境变量则输出毫米并按规范阈值分级。
"""
import os, sys, glob, json
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantify import quantify
from grade import grade
from train_seg import UNet
from joint_filter import reject_joints

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")
REAL = os.path.join(ROOT, "真实照片测试", "铁塔场景验证")
device = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
WORK_LONG = int(os.environ.get("WORK_LONG", 480))
THR = float(os.environ.get("SEG_TH", 0.5))
MM_PER_PX = os.environ.get("MM_PER_PX")
MM_PER_PX = float(MM_PER_PX) if MM_PER_PX else None
DAM = os.environ.get("DAM_MODE", "0") == "1"
JOINT_FILTER = os.environ.get("JOINT_FILTER", "0") == "1"   # 接缝几何后处理(默认开)

def load_model():
    ck = torch.load(f"{OUT}/best_seg.pth", map_location=device, weights_only=False)
    m = UNet().to(device); m.load_state_dict(ck["state_dict"]); m.eval()
    return m

def _norm(img_rgb):
    return ((img_rgb.astype(np.float32)/255.0 - MEAN)/STD).transpose(2,0,1)

@torch.no_grad()
def seg_batch(model, imgs_rgb):
    """一批同尺寸小图 -> 概率图列表（尺寸需为16的倍数，不足则pad）。"""
    H,W = imgs_rgb[0].shape[:2]
    ph, pw = (-H) % 16, (-W) % 16
    x = np.stack([_norm(np.pad(i, ((0,ph),(0,pw),(0,0)), mode="reflect")) for i in imgs_rgb])
    p = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()[:,0]
    return [q[:H,:W] for q in p]

@torch.no_grad()
def seg_large(model, img_rgb, win=192, stride=96):
    """大图滑窗分割，重叠区取平均。"""
    H,W = img_rgb.shape[:2]
    if H < win or W < win:
        return seg_batch(model,[img_rgb])[0]
    acc = np.zeros((H,W), np.float32); cnt = np.zeros((H,W), np.float32)
    ys = list(range(0,H-win+1,stride)) + ([H-win] if (H-win) % stride else [])
    xs = list(range(0,W-win+1,stride)) + ([W-win] if (W-win) % stride else [])
    tiles, pos = [], []
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            tiles.append(img_rgb[y:y+win, x:x+win]); pos.append((y,x))
    for i in range(0, len(tiles), 32):
        probs = seg_batch(model, tiles[i:i+32])
        for (y,x), pr in zip(pos[i:i+32], probs):
            acc[y:y+win, x:x+win] += pr; cnt[y:y+win, x:x+win] += 1
    return acc/np.maximum(cnt,1)

def fit_scale(img, long_side=WORK_LONG):
    h,w = img.shape[:2]; s = long_side/max(h,w)
    return cv2.resize(img, (max(16,int(round(w*s))), max(16,int(round(h*s))))) if s < 1 or s > 1 else img

def analyze(model, img_bgr, large=True):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = fit_scale(rgb)
    prob = seg_large(model, rgb) if large else seg_batch(model,[rgb])[0]
    mask = prob >= THR
    if JOINT_FILTER:
        mask = reject_joints(mask, image=rgb)   # 原图上识别规则接缝并剔除，保留裂缝
    q = quantify(mask, mm_per_px=MM_PER_PX)
    g = grade(q, dam_mode=DAM)
    return rgb, mask, q, g

def overlay(rgb, mask, g, path):
    vis = rgb.copy()
    vis[mask] = (0.35*vis[mask] + 0.65*np.array([255,40,40])).astype(np.uint8)
    col = {0:(90,200,130),1:(220,190,60),2:(230,140,60),3:(220,60,60)}[g["level"]]
    bar = np.full((26, vis.shape[1], 3), 30, np.uint8)
    vis = np.vstack([bar, vis])
    cv2.rectangle(vis,(0,0),(vis.shape[1],26),(30,30,34),-1)
    cv2.putText(vis, f"LV{g['level']} {g['width_used']}{g['unit']}", (6,19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col[::-1], 2)
    cv2.imwrite(path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def run_real(model):
    files = sorted(f for f in glob.glob(f"{REAL}/*.jpg")
                   if all(k not in f for k in ("_检测结果","对比","contact","画廊")))
    os.makedirs(f"{OUT}/auto_real", exist_ok=True)
    res={}
    for f in files:
        rgb, mask, q, g = analyze(model, cv2.imread(f), large=True)
        n = os.path.basename(f)
        overlay(rgb, mask, g, f"{OUT}/auto_real/{n[:-4]}_LV{g['level']}.jpg")
        res[n] = {"grade":g, "q":{k:q[k] for k in
                  ["area_ratio","p95_width_px","max_width_px","length_px",
                   "n_components","n_branch","crack_type","orientation_deg"]}}
        print(f"  LV{g['level']} {n:16s} w={g['width_used']}{g['unit']:<3s} {q['crack_type']:8s} {g['reasons'][0]}")
    json.dump(res, open(f"{OUT}/auto_grade_real.json","w"), ensure_ascii=False, indent=1)
    print(f"\n{len(res)} 张真实照片 -> outputs/auto_grade_real.json")
    return res

def run_crackforest(model, limit=None):
    files=[]
    for sp in ["train","val","test"]:
        for c in ["crack","nocrack"]:
            files += [(p,c) for p in sorted(glob.glob(f"{ROOT}/data/{sp}/{c}/*.jpg"))]
    if limit: files = files[:limit]
    print(f"CrackForest 数据样例 {len(files)} 张，批量分级中…")
    res={}; B=64
    for i in range(0, len(files), B):
        chunk = files[i:i+B]
        imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p,_ in chunk]
        probs = seg_batch(model, imgs)
        for (p,c), pr in zip(chunk, probs):
            q = quantify(pr >= THR, mm_per_px=MM_PER_PX); g = grade(q, dam_mode=DAM)
            res[os.path.basename(p)] = {"truth":c, "level":g["level"],
                "width":g["width_used"], "unit":g["unit"], "type":q["crack_type"],
                "area_ratio":round(q["area_ratio"],5)}
        if (i//B) % 10 == 0: print(f"  …{min(i+B,len(files))}/{len(files)}")
    json.dump(res, open(f"{OUT}/auto_grade_crackforest.json","w"), ensure_ascii=False)
    from collections import Counter
    for c in ["crack","nocrack"]:
        cnt = Counter(v["level"] for v in res.values() if v["truth"]==c)
        tot = sum(cnt.values())
        print(f"  {c:8s} n={tot:5d} " + "  ".join(f"LV{k}:{cnt.get(k,0)}({cnt.get(k,0)/max(tot,1)*100:.1f}%)" for k in range(4)))
    print(f"\n{len(res)} 张 -> outputs/auto_grade_crackforest.json")
    return res

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv)>1 else "real"
    model = load_model()
    print(f"设备 {device} | 分割阈值 {THR} | 工作长边 {WORK_LONG}px | "
          f"标定 {'%.4f mm/px'%MM_PER_PX if MM_PER_PX else '未标定(像素相对分级)'} | 堤坝模式 {DAM}")
    if mode == "real": run_real(model)
    elif mode == "crackforest":
        run_crackforest(model, int(sys.argv[2]) if len(sys.argv)>2 else None)
    elif mode == "file":
        for f in sys.argv[2:]:
            rgb, mask, q, g = analyze(model, cv2.imread(f))
            print(f"{os.path.basename(f)}: LV{g['level']} {g['name']} | {g['width_used']}{g['unit']} | {q['crack_type']}")
            for r in g["reasons"]: print("   -", r)
