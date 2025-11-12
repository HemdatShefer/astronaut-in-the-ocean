#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Photo detector v6
- Fast, single-image detector for boats/ships and persons.
- Adds boat subtypes (warship, military vessel, submarine, kayak, etc.) via CLIP/OpenCLIP.
- Multi-scale + tiling + flip TTA + horizon-tiny pass + wake (foam trail) proposals.
- Prioritized subtype de-duplication: remove overlapping subtype boxes and keep the best one.
- Heuristic+CLIP upgrade from generic 'boat' to a stronger subtype when geometry/textures match.
- Strong post-processing: WBF -> Soft-NMS -> Hard-NMS -> containment suppression.
Outputs:
  - /<save_dir>/vis/<image> — visualized PNG/JPG with boxes
  - /<save_dir>/detections.csv — per-image detections (class, subtype, confidence, bbox, clip_delta)
"""

import os, cv2, csv, math, argparse, glob, traceback
from pathlib import Path
import numpy as np
import torch


# ================== CONFIG ==================
class Cfg:
    # Runtime / speeds
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FP16   = True                    # use half precision on GPU (if available)
    IMGSZ  = 1440                   # network inference size (rounded up to /32)
    MIN_CONF = 0.07                 # minimum confidence for raw detections
    FINAL_NMS = 0.55                # IoU for final NMS (larger = more suppression)
    IOU_FUSE  = 0.42                # IoU for simple WBF-style box fusion

    # Image pyramid and tiling (helps far/small objects while keeping speed)
    PYRAMID = [1.0, 2.0, 3.0]       # scales for whole-frame passes
    TILE_W = 320                    # column-wise tiles (width in px on original image)
    TILE_OVERLAP = 0.35             # fractional horizontal overlap between tiles
    FLIP_TTA = True                 # horizontal flip TTA

    # Extra tiny pass around the horizon (many distant ships live there)
    H_TINY = True
    H_Y_FRAC = 0.45                 # top stripe height (fraction of H) for horizon band
    H_SCALE  = 3.2                  # upsample factor when scanning the horizon stripe
    H_TILE_W = 224                  # narrow tiles within the horizon stripe
    H_TILE_OVERLAP = 0.40

    # Simple enhancement (CLAHE + sharpening) for low-contrast water/sky
    ENHANCE = True

    # Water ROI: keep boats only if enough area falls inside [ROI_Y1, ROI_Y2]
    ROI_Y1 = 0.03                   # lower bound (fraction of height)
    ROI_Y2 = 1.00                   # upper bound (fraction of height)
    ROI_KEEP_FRAC = 0.03            # required ROI fraction inside a boat bbox

    # Wake proposals (derive candidate boxes from bright foam trails)
    WAKE = True
    WAKE_MIN_LEN = 90               # Hough min line length
    WAKE_PAD      = 26              # padding around the wake peak to form a bbox
    WAKE_CONF     = 0.22            # pseudo-confidence for wake proposals
    WAKE_EDGE_MIN = 0.10            # min edge density in the proposal crop
    WAKE_USE_CLIP = True            # validate wake proposals with CLIP
    WAKE_CLIP_DELTA = 0.07          # minimal CLIP (label - background) delta for a keep

    # CLIP gates
    USE_CLIP = False
    CLIP_THRESH = 0.07              # accept subtype if (label_sim - bg_sim) >= this
    CLIP_BG_GATE = 0.04             # drop smooth big sea patches with delta < this

    # Upgrade generic 'boat' -> subtype based on CLIP + geometry
    UPG_MIN_DELTA = 0.06            # (unused directly; kept for readability)
    UPG_RELAX_DELTA = 0.03          # allow upgrade with weaker CLIP if geometry screams "military"
    UPG_AREA = 120000               # consider "large" ship from this area (px^2)
    UPG_AR_HI = 0.65                # h/w ratio threshold for tall superstructures
    UPG_VERT_EDGE = 1.25            # mean|SobelX| / mean|SobelY| threshold for vertical structures

    # CLIP/OpenCLIP prompts for each subtype
    BOAT_SUBTYPE_PROMPTS = {
        "submarine": ["a submarine","a navy submarine","a military submarine"],
        "warship": ["a warship","a navy warship","a destroyer","a frigate","a military ship"],
        "military vessel": ["a military vessel","a navy vessel"],
        "patrol boat": ["a patrol boat","a navy patrol boat"],
        "coast guard ship": ["a coast guard ship","coast guard vessel"],
        "cargo ship": ["a cargo ship","a freighter"],
        "container ship": ["a container ship"],
        "ferry": ["a ferry"],
        "tugboat": ["a tugboat"],
        "yacht": ["a yacht"],
        "sailboat": ["a sailboat","a sailing boat"],
        "speedboat": ["a speedboat","a motorboat"],
        "kayak": ["a kayak"],
        "dinghy": ["a dinghy"],
        "jet ski": ["a jet ski"],
        "vessel": ["a vessel","a boat"]
    }

    # Subtype priority (used when overlapping subtypes compete)
    SUBTYPE_PRIORITY = {
        "warship": 7,
        "military vessel": 6,
        "submarine": 6,
        "coast guard ship": 5,
        "patrol boat": 5,
        "container ship": 4,
        "cargo ship": 4,
        "ferry": 4,
        "tugboat": 3,
        "speedboat": 3,
        "yacht": 3,
        "sailboat": 3,
        "jet ski": 2,
        "kayak": 2,
        "dinghy": 2,
        "vessel": 1,
        "boat": 0
    }

# Enable cuDNN autotuner and fast matmul for speed on CUDA
torch.backends.cudnn.benchmark = True
if Cfg.DEVICE == "cuda":
    try: torch.set_float32_matmul_precision("high")
    except: pass

# ================== UTILS ==================
def _round_imgsz(n):
    """Round a scalar image size up to the nearest multiple of 32 (as many backbones require)."""
    return int(max(32, math.ceil(float(n)/32.0)*32))

def _iou(a,b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1=max(a[0],b[0]); y1=max(a[1],b[1])
    x2=min(a[2],b[2]); y2=min(a[3],b[3])
    iw=max(0.0,x2-x1); ih=max(0.0,y2-y1)
    inter=iw*ih
    if inter<=0: return 0.0
    Sa=max(0.0,a[2]-a[0])*max(0.0,a[3]-a[1])
    Sb=max(0.0,b[2]-b[0])*max(0.0,b[3]-b[1])
    return float(inter/(Sa+Sb-inter+1e-6))

def _contains(big, small, thr=0.90):
    """Return True if `big` contains at least `thr` fraction of `small`'s area."""
    x1=max(big[0], small[0]); y1=max(big[1], small[1])
    x2=min(big[2], small[2]); y2=min(big[3], small[3])
    iw=max(0.0,x2-x1); ih=max(0.0,y2-y1)
    inter=iw*ih
    sw=max(0.0, small[2]-small[0]); sh=max(0.0, small[3]-small[1])
    S=sw*sh
    return (S>0) and (inter/S >= thr)

def soft_nms(dets, iou_thr=0.55, sigma=0.5, conf_thr=0.03):
    """
    Class-aware Soft-NMS:
    - Keep top-scoring box,
    - Reduce scores of same-class boxes by a Gaussian of IoU,
    - Drop if score falls below `conf_thr`.
    """
    if not dets: return []
    dets=sorted(dets, key=lambda d:d["conf"], reverse=True)
    keep=[]
    while dets:
        a=dets[0]; keep.append(a)
        rest=[]
        for d in dets[1:]:
            if d["cls"]!=a["cls"]:
                rest.append(d); continue
            i=_iou(a["bbox"], d["bbox"])
            d2=dict(d); d2["conf"]=d["conf"]*math.exp(-(i*i)/sigma)
            if d2["conf"]>=conf_thr: rest.append(d2)
        dets=sorted(rest, key=lambda d:d["conf"], reverse=True)
    return keep

def hard_nms(dets, iou_thr=0.55):
    """Classic greedy NMS per class."""
    if not dets: return []
    dets=sorted(dets, key=lambda d:d["conf"], reverse=True)
    keep=[]
    while dets:
        a=dets.pop(0); keep.append(a)
        dets=[d for d in dets if _iou(d["bbox"], a["bbox"]) < iou_thr]
    return keep

def fuse_wbf(dets, iou_thr=0.42):
    """
    Simple class-aware WBF-like fusion:
    - Cluster boxes by IoU >= iou_thr
    - Weighted-average coordinates by scores, keep max score.
    """
    if not dets: return []
    dets=sorted(dets, key=lambda d:d["conf"], reverse=True)
    used=[False]*len(dets); out=[]
    for i,a in enumerate(dets):
        if used[i]: continue
        cluster=[i]
        for j in range(i+1,len(dets)):
            if used[j]: continue
            if dets[j]["cls"]!=a["cls"]: continue
            if _iou(a["bbox"], dets[j]["bbox"])>=iou_thr:
                used[j]=True; cluster.append(j)
        if len(cluster)==1:
            out.append(a); continue
        boxes=np.array([dets[k]["bbox"] for k in cluster], np.float32)
        scores=np.array([dets[k]["conf"] for k in cluster], np.float32)
        w=scores/(scores.sum()+1e-6)
        box=(boxes*w[:,None]).sum(0).tolist()
        out.append({"bbox":box,"conf":float(scores.max()),"cls":a["cls"]})
    return out

def _auto_font(h):
    """Adaptive overlay font size & thickness based on image height."""
    fs=max(0.4, h*0.0009)
    th=max(1, int(round(h*0.0015)))
    return fs, th

def _badge(img, text, org, fs, th, alpha=0.55):
    """Draw a white translucent label paddle with black text."""
    x,y=org; font=cv2.FONT_HERSHEY_SIMPLEX
    (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
    pad=max(3,int(4*fs))
    x1=max(0,x-pad); y1=max(0,y-tht-pad); x2=x+tw+pad; y2=y+pad
    over=img.copy()
    cv2.rectangle(over,(x1,y1),(x2,y2),(255,255,255),-1)
    cv2.addWeighted(over, alpha, img, 1-alpha, 0, dst=img)
    cv2.putText(img,text,(x+1,y+1), font,fs,(255,255,255),th+2,cv2.LINE_AA)
    cv2.putText(img,text,(x,y),     font,fs,(0,0,0),      th,  cv2.LINE_AA)

def enhance(img):
    """Light CLAHE + unsharp mask to boost contrast and edges on water/sky."""
    if not Cfg.ENHANCE: return img
    lab=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab); l=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    lab=cv2.merge([l,a,b]); res=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur=cv2.GaussianBlur(res,(0,0),0.8); sharp=cv2.addWeighted(res,1.30,blur,-0.30,0)
    return sharp

# ================== YOLO ==================
# Normalize a range of ship-like label names to the 'boat' class.
_BOAT_ALIASES = {"boat","ship","vessel","yacht","sailboat","speedboat","ferry","barge","submarine","warship"}

def _post_ultra(result):
    """
    Normalize Ultralytics results into [{'bbox':[x1,y1,x2,y2], 'conf':float, 'cls':'boat'|'person'}].
    We treat any boat/ship/vessel label as 'boat' (subtypes are added later by CLIP).
    """
    out=[]
    if result is None or getattr(result,"boxes",None) is None: return out
    boxes=result.boxes
    xyxy=boxes.xyxy.float().cpu().numpy()
    conf=boxes.conf.float().cpu().numpy()
    cls =boxes.cls.int().cpu().numpy()
    names=getattr(result,'names',None)
    for i in range(len(xyxy)):
        cid=int(cls[i])
        name=str(names[cid]).lower() if (names and cid<len(names)) else str(cid)
        if name in _BOAT_ALIASES or name in ("8","9"):  # fallback numeric ids for some models
            out.append({'bbox':[float(v) for v in xyxy[i]], 'conf':float(conf[i]), 'cls':'boat'})
        elif name in ("person","0"):
            out.append({'bbox':[float(v) for v in xyxy[i]], 'conf':float(conf[i]), 'cls':'person'})
    return out

class YOLODetector:
    """Thin wrapper around Ultralytics YOLO to keep a normalized output format."""
    def __init__(self, weights):
        from ultralytics import YOLO
        self.ok=False
        try:
            self.model=YOLO(weights)
            # warmup to compile backends / allocate memory
            dummy=np.zeros((640,640,3), np.uint8)
            _=self.model.predict([dummy], imgsz=640, verbose=False,
                                 device=("cpu" if Cfg.DEVICE=="cpu" else 0),
                                 half=bool(Cfg.FP16 and torch.cuda.is_available()))
            self.ok=True; print(f"[YOLO] loaded: {weights}")
        except Exception as e:
            self.model=None; print(f"[YOLO] failed to load: {e}")
    def predict(self, frames, imgsz):
        if not self.ok or not frames: return [[] for _ in frames]
        ms=_round_imgsz(imgsz)
        r=self.model.predict(frames, imgsz=ms, verbose=False,
                             device=("cpu" if Cfg.DEVICE=='cpu' else 0),
                             half=bool(Cfg.FP16 and torch.cuda.is_available()))
        return [ _post_ultra(rr) for rr in r ]

# ============== CLIP / OpenCLIP ==============
class SubtypeClassifier:
    """
    Adds subtypes to generic 'boat' boxes using CLIP/OpenCLIP:
    - Builds a text embedding bank for each subtype + a 'background sea' concept.
    - For each crop: computes (similarity to label - similarity to background).
    - Returns top-2 labels and their deltas; caller applies thresholds/priorities.
    """
    def __init__(self, label_prompts: dict):
        self.ok=False
        self.labels=list(label_prompts.keys())
        self.is_openclip=False
        self.preprocess=None
        self.bg_index=None
        try:
            import open_clip
            self.is_openclip=True
            self.open_clip=open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device=Cfg.DEVICE)
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.ok=True; print("[OpenCLIP] ViT-B/32 loaded")
        except Exception:
            try:
                import clip
                self.clip=clip
                self.model, self.preprocess = clip.load("ViT-B/32", device=Cfg.DEVICE, jit=False)
                self.ok=True; print("[CLIP] ViT-B/32 loaded")
            except Exception as e:
                print(f"[CLIP] unavailable ({e}) — heuristics only")
                self.ok=False

        # Pre-compute normalized text embeddings (labels + background).
        self.text_norm=None
        if self.ok:
            with torch.no_grad():
                embs=[]
                for lab in self.labels:
                    prompts=[f"a photo of a {p}" for p in label_prompts[lab]]
                    if self.is_openclip:
                        tok=self.tokenizer(prompts).to(Cfg.DEVICE)
                        txt=self.model.encode_text(tok).float()
                    else:
                        tok=self.clip.tokenize(prompts).to(Cfg.DEVICE)
                        txt=self.model.encode_text(tok).float()
                    txt=txt/(txt.norm(dim=-1, keepdim=True)+1e-6)
                    embs.append(txt.mean(dim=0, keepdim=True))
                bgp=["empty sea","open water with no boats","waves only"]
                if self.is_openclip:
                    tbg=self.model.encode_text(self.tokenizer(bgp).to(Cfg.DEVICE)).float()
                else:
                    tbg=self.model.encode_text(self.clip.tokenize(bgp).to(Cfg.DEVICE)).float()
                tbg=tbg/(tbg.norm(dim=-1, keepdim=True)+1e-6)
                self.text_norm=torch.cat(embs+[tbg.mean(dim=0,keepdim=True)], dim=0)
                self.bg_index=len(self.labels)

    @torch.no_grad()
    def classify_top2(self, frame_bgr, boxes):
        """Return (best_label, best_delta, second_label, second_delta) for each box."""
        K=len(self.labels)
        out=[]
        if not self.ok or not boxes:
            for _ in boxes: out.append(("boat",0.0,"boat",0.0))
            return out
        from PIL import Image
        H,W=frame_bgr.shape[:2]
        crops=[]
        for b in boxes:
            x1,y1,x2,y2=map(int,[max(0,b[0]),max(0,b[1]),min(W-1,b[2]),min(H-1,b[3])])
            if x2-x1<2 or y2-y1<2: crops.append(None); continue
            crop=Image.fromarray(cv2.cvtColor(frame_bgr[y1:y2,x1:x2], cv2.COLOR_BGR2RGB))
            crops.append(self.preprocess(crop).unsqueeze(0))
        X=torch.cat([c for c in crops if c is not None],dim=0).to(Cfg.DEVICE) if any(c is not None for c in crops) else None
        if X is None:
            for _ in boxes: out.append(("boat",0.0,"boat",0.0))
            return out
        feats=self.model.encode_image(X).float()
        feats=feats/(feats.norm(dim=-1, keepdim=True)+1e-6)
        sims=feats @ self.text_norm.T
        vi=0
        for c in crops:
            if c is None:
                out.append(("boat",0.0,"boat",0.0)); continue
            s=sims[vi,:K]; sbg=float(sims[vi,self.bg_index]); vi+=1
            vals=s.tolist()
            j1=int(np.argmax(vals)); v1=vals[j1]-sbg
            vals[j1]=-1e9
            j2=int(np.argmax(vals)); v2=vals[j2]-sbg
            out.append((self.labels[j1], float(v1), self.labels[j2], float(v2)))
        return out

# ================== IO ==================
IMG_EXTS={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
def list_images(src):
    """Resolve a file/dir/glob path into a sorted list of image paths."""
    p=Path(src)
    if p.is_dir():
        files=[]
        for ext in IMG_EXTS:
            files+=sorted(p.rglob(f"*{ext}"))
        return [str(f) for f in files]
    if p.is_file(): return [str(p)]
    out=[]
    for f in glob.glob(src):
        if Path(f).suffix.lower() in IMG_EXTS: out.append(f)
    return sorted(out)

# ================== ROI ==================
def roi_mask(h,w,y1_fr,y2_fr):
    """Binary mask for [y1_fr, y2_fr] band (used to keep boats on water only)."""
    m=np.zeros((h,w), np.uint8)
    y1=int(h*max(0.0,min(1.0,y1_fr))); y2=int(h*max(0.0,min(1.0,y2_fr)))
    y1=min(y1,y2); y2=max(y1+1,y2)
    m[y1:y2,:]=1
    return m

def filter_by_roi(dets, mask, keep_frac=0.03):
    """Keep boat detections only if enough bbox area intersects the water ROI."""
    if mask is None: return dets
    h,w=mask.shape; out=[]
    for d in dets:
        x1,y1,x2,y2=map(int,[max(0,d["bbox"][0]),max(0,d["bbox"][1]),
                              min(w-1,d["bbox"][2]),min(h-1,d["bbox"][3])])
        if x2<=x1 or y2<=y1: continue
        roi=mask[y1:y2,x1:x2]; frac=float(roi.mean()) if roi.size else 0.0
        if d["cls"]=="boat":
            if frac>=keep_frac: out.append(d)
        else:
            out.append(d)
    return out

# ================== VIS ==================
def draw_det(img, dets):
    """Draw rectangles and label paddles for both persons and boats (with subtype)."""
    h,w=img.shape[:2]; fs,th=_auto_font(h); a=0.55
    orange=(0,140,255); green=(0,200,0)
    for d in dets:
        x1,y1,x2,y2=map(int,d["bbox"])
        if d["cls"]=="person":
            cv2.rectangle(img,(x1,y1),(x2,y2),green,max(2,th),cv2.LINE_AA)
            _badge(img, f"person {d['conf']:.2f}", (x1,max(y1-6,15)), fs, th, a)
        else:
            cv2.rectangle(img,(x1,y1),(x2,y2),orange,max(2,th),cv2.LINE_AA)
            name = d.get("subtype","boat")
            _badge(img, f"{name} {d['conf']:.2f}", (x1,max(y1-6,15)), fs, th, a)

# ================== WAKE PROPOSALS ==================
def wake_proposals(img, clip_model=None):
    """
    Propose boat boxes around bright foam trails:
    - Canny + HoughLinesP to find long linear wakes,
    - Expand around the bright "head" of the wake,
    - Check aspect & edge density,
    - Optionally validate with CLIP delta vs background.
    """
    if not Cfg.WAKE: return []
    H,W=img.shape[:2]
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),1.2)
    edges=cv2.Canny(blur,50,130)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=70,
                          minLineLength=Cfg.WAKE_MIN_LEN, maxLineGap=10)
    props=[]
    if lines is None: return props
    edge_map=edges>0
    for l in lines[:,0,:]:
        x1,y1,x2,y2=l.tolist()
        sx,sy=(x1,y1) if y1>y2 else (x2,y2)  # pick the lower end as the likely boat head
        r=Cfg.WAKE_PAD
        xA=max(0,sx-r); yA=max(0,sy-r); xB=min(W,sx+r); yB=min(H,sy+r)
        if xB-xA<12 or yB-yA<12: continue
        crop=gray[yA:yB,xA:xB]
        _,th=cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        inv=cv2.bitwise_not(th)
        cnts,_=cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c=max(cnts, key=cv2.contourArea)
        x,y,w,h=cv2.boundingRect(c)
        bx1=max(0,xA+x- max(12,int(w*1.5)))
        by1=max(0,yA+y- max(8,int(h*1.2)))
        bx2=min(W,xA+x+w+ max(12,int(w*1.5)))
        by2=min(H,yA+y+h+ max(8,int(h*1.2)))
        bw=max(1,bx2-bx1); bh=max(1,by2-by1)
        ar=bw/float(bh)
        if ar<2.2:  # too boxy (not an elongated boat+wake)
            continue
        ed = float(edge_map[by1:by2, bx1:bx2].mean())
        if ed < Cfg.WAKE_EDGE_MIN:
            continue
        cand={"bbox":[float(bx1),float(by1),float(bx2),float(by2)],
              "conf":float(Cfg.WAKE_CONF), "cls":"boat"}
        if Cfg.WAKE_USE_CLIP and clip_model is not None and clip_model.ok:
            (lab, delta, _, _) = clip_model.classify_top2(img, [[bx1,by1,bx2,by2]])[0]
            if delta < Cfg.WAKE_CLIP_DELTA:
                continue
        props.append(cand)
    return props

# ================== HORIZON TINY PASS ==================
def horizon_tiny_pass(det, img):
    """
    Scan only the top stripe (horizon band), upscaled and in narrow tiles.
    This is cheap and finds tiny far-away ships without scanning the whole frame.
    """
    if not Cfg.H_TINY: return []
    H,W=img.shape[:2]
    y2=int(H*max(0.1,min(0.95,Cfg.H_Y_FRAC)))
    stripe=img[:y2,:]
    s=Cfg.H_SCALE if Cfg.H_SCALE>1.0 else 1.0
    up=cv2.resize(stripe,(int(W*s), int(y2*s)), interpolation=cv2.INTER_CUBIC)
    outs=[]
    # full horizon pass
    o=det.predict([up], Cfg.IMGSZ)[0]
    for d in o:
        x1,y1,x2,y2=d["bbox"]
        outs.append({"bbox":[x1/s, y1/s, x2/s, y2/s], "conf":d["conf"], "cls":d["cls"]})
    # tiled horizon pass
    tw=int(Cfg.H_TILE_W); step=int(tw*(1.0-Cfg.H_TILE_OVERLAP))
    for x0 in range(0, up.shape[1], max(1,step)):
        x1=min(up.shape[1], x0+tw)
        if x1-x0<tw//2: break
        crop=up[:, x0:x1]
        o=det.predict([crop], Cfg.IMGSZ)[0]
        for d in o:
            xa,ya,xb,yb=d["bbox"]
            outs.append({"bbox":[(xa+x0)/s, ya/s, (xb+x0)/s, yb/s], "conf":d["conf"], "cls":d["cls"]})
    # keep only boxes entirely in the stripe (avoid duplicates from main pass)
    outs=[d for d in outs if d["bbox"][3] <= y2+6]
    return outs

# ======== CLIP gate & subtype logic / dedupe ========
def clip_gate_boats(dets):
    """
    Drop generic 'boat' boxes likely to be empty water:
    if CLIP (label-bg) delta is very small and the bbox is large & smooth.
    """
    out=[]
    for d in dets:
        if d["cls"]!="boat": out.append(d); continue
        delta=float(d.get("clip_delta", 1.0))
        if delta < Cfg.CLIP_BG_GATE:
            x1,y1,x2,y2=map(int,d["bbox"])
            if (x2-x1)*(y2-y1) > 9000:  # big smooth area — drop
                continue
        out.append(d)
    return out

def subtype_priority(name):
    """Map subtype name to numeric priority."""
    return Cfg.SUBTYPE_PRIORITY.get(name, 0)

def prune_cross_subtypes(dets, iou_thr=0.60):
    """
    If the same physical object received multiple subtype labels
    (high IoU between boxes), keep only one:
      - prefer higher subtype priority, then higher confidence.
    """
    boats=[d for d in dets if d["cls"]=="boat"]
    others=[d for d in dets if d["cls"]!="boat"]
    boats=sorted(boats, key=lambda d:(-subtype_priority(d.get("subtype","boat")), -d["conf"]))
    used=[False]*len(boats); keep=[]
    for i,a in enumerate(boats):
        if used[i]: continue
        used[i]=True; keep.append(a)
        for j in range(i+1,len(boats)):
            if used[j]: continue
            b=boats[j]
            if _iou(a["bbox"], b["bbox"])>=iou_thr:
                # suppress the lower-priority/overlapping candidate
                used[j]=True
    return keep+others

def upgrade_generic_boats(img, dets, subtype: "SubtypeClassifier"):
    """
    Upgrade 'boat' -> (warship|military vessel|...) using:
      - CLIP/OpenCLIP (best_label - background) delta,
      - Geometry (tall superstructure, large area),
      - Vertical edge dominance (masts/turrets).
    """
    boats_idx=[i for i,d in enumerate(dets) if d["cls"]=="boat"]
    if not boats_idx: return dets
    if subtype is None or not subtype.ok:
        # Fallback: geometry-only upgrade
        for i in boats_idx:
            x1,y1,x2,y2=map(int,dets[i]["bbox"])
            w=max(1,x2-x1); h=max(1,y2-y1); area=w*h
            ar=h/float(w)
            if area>=Cfg.UPG_AREA and ar>=Cfg.UPG_AR_HI:
                dets[i]["subtype"]="military vessel"
        return dets

    # CLIP top-2 scores & deltas
    info = subtype.classify_top2(img, [dets[i]["bbox"] for i in boats_idx])
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for (idx,(best,bd,sec,sd)) in zip(boats_idx, info):
        d=dets[idx]
        x1,y1,x2,y2=map(int,d["bbox"])
        w=max(1,x2-x1); h=max(1,y2-y1); area=w*h
        crop=gray[y1:y2,x1:x2]
        if crop.size<4:
            d["subtype"]=best if bd>=Cfg.CLIP_THRESH else "boat"; d["clip_delta"]=bd; continue
        # vertical texture strength
        sx=cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
        sy=cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
        v_ratio=(np.mean(np.abs(sx))+1e-6)/(np.mean(np.abs(sy))+1e-6)
        tall = (h/float(w) >= Cfg.UPG_AR_HI)
        big  = (area >= Cfg.UPG_AREA)
        strong_label = best in ("warship","military vessel")
        relax = (bd>=Cfg.UPG_RELAX_DELTA and strong_label and ( (tall and big) or v_ratio>=Cfg.UPG_VERT_EDGE ))
        if bd>=Cfg.CLIP_THRESH or relax:
            d["subtype"]=best
        else:
            d["subtype"]="boat"
        d["clip_delta"]=float(bd)
    return dets

def prefer_subtype_over_generic(dets):
    """Drop generic 'boat' if a subtype box overlaps strongly with it (IoU >= 0.60)."""
    boats=[d for d in dets if d["cls"]=="boat"]
    persons=[d for d in dets if d["cls"]=="person"]
    generics=[d for d in boats if d.get("subtype","boat")=="boat"]
    subs    =[d for d in boats if d.get("subtype","boat")!="boat"]
    keep=subs[:]
    for g in generics:
        drop=False
        for s in subs:
            print(_iou(g["bbox"], s["bbox"]))
            if _iou(g["bbox"], s["bbox"])>=0.60:
                drop=True; break
        if not drop: keep.append(g)
    return keep+persons

def suppress_contained(dets):
    """Remove near-contained duplicates (>=90% area containment) within the same class."""
    dets=sorted(dets, key=lambda d:(d["cls"], -d["conf"], -(d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1])))
    used=[False]*len(dets); out=[]
    for i,a in enumerate(dets):
        if used[i]: continue
        for j in range(i+1,len(dets)):
            if used[j]: continue
            b=dets[j]
            if a["cls"]!=b["cls"]: continue
            if _contains(a["bbox"], b["bbox"], thr=0.90): used[j]=True
            elif _contains(b["bbox"], a["bbox"], thr=0.90): used[i]=True; break
        if not used[i]: out.append(a)
    return out

# ================== STRONG DETECT ==================
def detect_strong(det: "YOLODetector", img, subtype: "SubtypeClassifier"):
    """
    Multi-pass detection that trades a bit of compute for quality on small/far boats:
      - Full-frame at base size, plus extra pyramid scales,
      - Column-wise tiling,
      - Flip TTA,
      - Horizon-only tiny pass,
      - Wake-based proposals,
      - WBF + Soft-NMS + Hard-NMS cascade.
    """
    H,W=img.shape[:2]
    all_d=[]
    # base pass
    all_d += det.predict([img], Cfg.IMGSZ)[0]
    # pyramid
    for s in Cfg.PYRAMID:
        if abs(s-1.0)<1e-6: continue
        f=cv2.resize(img,(int(W*s),int(H*s)), interpolation=cv2.INTER_LINEAR)
        o=det.predict([f], Cfg.IMGSZ)[0]
        for d in o:
            x1,y1,x2,y2=d["bbox"]
            all_d.append({"bbox":[x1/s,y1/s,x2/s,y2/s], "conf":d["conf"], "cls":d["cls"]})
    # tiling
    tw=int(Cfg.TILE_W)
    if tw>0 and tw<W*0.95:
        step=int(tw*(1.0-Cfg.TILE_OVERLAP))
        for x0 in range(0, W, max(1,step)):
            x1=min(W, x0+tw)
            if x1-x0<tw//2: break
            crop=img[:, x0:x1]
            o=det.predict([crop], Cfg.IMGSZ)[0]
            for d in o:
                xa,ya,xb,yb=d["bbox"]
                all_d.append({"bbox":[xa+x0,ya,xb+x0,yb], "conf":d["conf"], "cls":d["cls"]})
    # flip TTA
    if Cfg.FLIP_TTA:
        f=cv2.flip(img,1)
        o=det.predict([f], Cfg.IMGSZ)[0]
        for d in o:
            x1,y1,x2,y2=d["bbox"]
            all_d.append({"bbox":[W-x2,y1,W-x1,y2], "conf":d["conf"], "cls":d["cls"]})
    # horizon + wakes
    all_d += horizon_tiny_pass(det, img)
    all_d += wake_proposals(img, subtype)

    # class filter and early conf cut
    all_d=[d for d in all_d if d["conf"]>=Cfg.MIN_CONF and d["cls"] in ("boat","person")]

    # fusion & suppression chain
    all_d=fuse_wbf(all_d, iou_thr=Cfg.IOU_FUSE)
    all_d=soft_nms(all_d, iou_thr=Cfg.FINAL_NMS)
    all_d=hard_nms(all_d, iou_thr=Cfg.FINAL_NMS)
    return all_d

# ================== MAIN ==================
def build_argparser():
    ap=argparse.ArgumentParser("Photo detector v6: person + boat subtypes, cross-subtype dedupe & upgrade")
    ap.add_argument("--source", required=True, help="Path to an image file, directory, or glob (e.g. /data/*.jpg)")
    ap.add_argument("--save_dir", required=True, help="Output directory (CSV + visualizations)")
    ap.add_argument("--yolo_weights", required=True, help="Ultralytics .pt weights")
    ap.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=Cfg.IMGSZ, help="Base inference size (rounded to /32)")
    ap.add_argument("--min_conf", type=float, default=Cfg.MIN_CONF, help="Min confidence for raw detections")
    ap.add_argument("--final_nms", type=float, default=Cfg.FINAL_NMS, help="IoU threshold for NMS")
    ap.add_argument("--pyramid", type=str, default="1.0,2.0,3.0", help="Comma-separated full-frame scales")
    ap.add_argument("--tile_w", type=int, default=Cfg.TILE_W, help="Column tile width on original image")
    ap.add_argument("--tile_overlap", type=float, default=Cfg.TILE_OVERLAP, help="Tile overlap fraction")
    ap.add_argument("--flip_tta", action="store_true", help="Enable horizontal flip TTA")
    ap.add_argument("--no_flip_tta", action="store_true", help="Disable flip TTA")
    ap.add_argument("--roi_y", type=float, nargs=2, default=[Cfg.ROI_Y1,Cfg.ROI_Y2], help="Water ROI [y1_frac y2_frac]")
    ap.add_argument("--roi_keep", type=float, default=Cfg.ROI_KEEP_FRAC, help="Boat bbox ROI intersection fraction")
    ap.add_argument("--no_clip", action="store_true", help="Disable CLIP/OpenCLIP subtype logic")
    ap.add_argument("--no_enhance", action="store_true", help="Disable CLAHE/sharpening")
    ap.add_argument("--h_tiny", type=int, default=1, help="1 to enable horizon tiny pass, 0 to disable")
    ap.add_argument("--h_y", type=float, default=Cfg.H_Y_FRAC, help="Horizon stripe height fraction")
    ap.add_argument("--h_scale", type=float, default=Cfg.H_SCALE, help="Upscale factor for horizon stripe")
    ap.add_argument("--h_tile_w", type=int, default=Cfg.H_TILE_W, help="Tile width for horizon stripe")
    ap.add_argument("--h_overlap", type=float, default=Cfg.H_TILE_OVERLAP, help="Overlap for horizon tiles")
    # wake
    ap.add_argument("--wake", type=int, default=1, help="1 to enable wake proposals")
    ap.add_argument("--wake_conf", type=float, default=Cfg.WAKE_CONF, help="Pseudo-confidence for wake proposals")
    ap.add_argument("--wake_edge", type=float, default=Cfg.WAKE_EDGE_MIN, help="Min edge density in wake box")
    ap.add_argument("--wake_clip", type=int, default=1, help="1 to validate wake proposals with CLIP")
    ap.add_argument("--wake_delta", type=float, default=Cfg.WAKE_CLIP_DELTA, help="CLIP (label-bg) delta for wake keep")
    return ap


def temp2(det, img):
    if not det.ok: raise RuntimeError("YOLO failed to load")
    subtype = SubtypeClassifier(Cfg.BOAT_SUBTYPE_PROMPTS) if Cfg.USE_CLIP else None
    try:
        if Cfg.ENHANCE: img = enhance(img)

        dets = detect_strong(det, img, subtype)
        m = roi_mask(img.shape[0], img.shape[1], Cfg.ROI_Y1, Cfg.ROI_Y2)
        dets = filter_by_roi(dets, m, keep_frac=Cfg.ROI_KEEP_FRAC)

        # upgrade generic -> subtype
        dets = upgrade_generic_boats(img, dets, subtype)

        # remove empty-sea false positives
        if subtype is not None: dets = clip_gate_boats(dets)

        # subtype precedence and overlap cleanup
        dets = prefer_subtype_over_generic(dets)
        dets = prune_cross_subtypes(dets, iou_thr=0.60)
        dets = suppress_contained(dets)

        # final fusion & suppression
        dets = fuse_wbf(dets, iou_thr=Cfg.IOU_FUSE)
        dets = hard_nms(dets, iou_thr=Cfg.FINAL_NMS)

        print(dets)

    except Exception as e:
        traceback.print_exc()
