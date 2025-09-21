from __future__ import annotations
import argparse, logging, os, json
from typing import Tuple, Optional, List
import numpy as np
import cv2

try:
    import torch
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    torch = None

try:
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    TV_OK = True
except Exception:
    TV_OK = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("arbres_depth_guided")

def _parse_args():
    p = argparse.ArgumentParser(description="Depth-Guided Tree Height (axis + depth edges)")
    p.add_argument("--image", type=str, help="Path to input image")
    p.add_argument("--save-debug", action="store_true")
    p.add_argument("--max-size", type=int, default=1600)
    p.add_argument("--pixel-per-meter", type=float, default=None)
    p.add_argument("--use-depth", action="store_true", help="Enable depth-guided apex/base")
    p.add_argument("--depth-model", type=str, default="DPT_Large", help="MiDaS model: DPT_Large | DPT_Hybrid | MiDaS_small")
    return p.parse_args()

def select_image_file() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    try:
        root = tk.Tk(); root.withdraw()
        filetypes = [("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select image", filetypes=filetypes)
        root.destroy()
        return path or None
    except Exception:
        return None

def get_image_path(cli_path: Optional[str]) -> Optional[str]:
    if cli_path and os.path.isfile(cli_path):
        return cli_path
    return select_image_file()

def load_image(path: str, max_side: int = 1600):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None: raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    return img_bgr, img_rgb, H, W

def _area_ratio(mask: np.ndarray) -> float:
    mask_bin = mask > 0
    return float(mask_bin.sum()) / (mask_bin.shape[0] * mask_bin.shape[1])

def _largest_component(mask_bool: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    if num_labels <= 1: return mask_bool
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)
    return labels == idx

def create_vegetation_mask_by_color(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    lab_eq = cv2.merge([L, A, B])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(rgb_eq, cv2.COLOR_RGB2HSV)
    lower1 = np.array([25, 25, 35], dtype=np.uint8)
    upper1 = np.array([85, 255, 255], dtype=np.uint8)
    lower2 = np.array([35, 15, 20], dtype=np.uint8)
    upper2 = np.array([95, 255, 200], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    lower_sky = np.array([90, 0, 120], dtype=np.uint8)
    upper_sky = np.array([140, 80, 255], dtype=np.uint8)
    sky = cv2.inRange(hsv, lower_sky, upper_sky)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(sky))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask_bool = mask > 0
    mask_bool = _largest_component(mask_bool)
    if mask_bool.sum() == 0: return None
    return (mask_bool.astype(np.uint8) * 255)

_MASKRCNN = None
def _get_maskrcnn(device):
    global _MASKRCNN
    if _MASKRCNN is None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        _MASKRCNN = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
    return _MASKRCNN

def try_maskrcnn_segmentation(img_rgb: np.ndarray, device="cpu") -> Optional[np.ndarray]:
    if not (TORCH_OK and TV_OK): return None
    model = _get_maskrcnn(device)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    inp = torch.from_numpy(x).unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model(inp)[0]
    masks = out.get("masks", None)
    if masks is None or len(masks) == 0: return None
    best, best_area = None, 0
    for m in masks.squeeze(1).detach().cpu().numpy():
        mb = (m > 0.5)
        area = mb.sum()
        if area > best_area:
            best, best_area = mb, area
    if best is None: return None
    return (best.astype(np.uint8) * 255)

def create_fallback_mask(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    H, W = gray.shape
    MIN_BLOB_RATIO = 0.02
    mask = np.zeros_like(gray, dtype=np.uint8)
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= MIN_BLOB_RATIO * H * W:
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    if mask.sum() == 0: return None
    mask_bool = _largest_component(mask > 0)
    return (mask_bool.astype(np.uint8) * 255)

def refine_tree_mask(mask: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
    mask_bin = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k, iterations=1)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k, iterations=1)
    mask_bin = _largest_component(mask_bin > 0).astype(np.uint8)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k2, iterations=1)
    return (mask_bin * 255)

def _seg_color_at_scale(img_rgb: np.ndarray, scale: float) -> Optional[np.ndarray]:
    if abs(scale - 1.0) < 1e-3:
        return create_vegetation_mask_by_color(img_rgb)
    h, w = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    m = create_vegetation_mask_by_color(small)
    if m is None: return None
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

def _tta_union_color(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    masks: List[np.ndarray] = []
    for s in (0.75, 1.0, 1.25):
        m = _seg_color_at_scale(img_rgb, s)
        if m is not None:
            masks.append(m > 0)
    if not masks: return None
    u = np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8) * 255
    return u

def segment_tree_multi_approach(img_rgb: np.ndarray, device="cpu") -> np.ndarray:
    MIN_RATIO = 0.005
    candidates: list[tuple[str, np.ndarray]] = []
    try:
        m = _tta_union_color(img_rgb)
        if m is not None and _area_ratio(m) >= MIN_RATIO:
            candidates.append(("color_tta", m))
    except Exception as e:
        log.debug(f"Color TTA failed: {e}")
    if TORCH_OK and TV_OK:
        try:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = try_maskrcnn_segmentation(img_rgb, device=dev)
            if m is not None and _area_ratio(m) >= MIN_RATIO:
                candidates.append(("rcnn", m))
        except Exception as e:
            log.debug(f"RCNN failed: {e}")
    if not candidates:
        try:
            m = create_fallback_mask(img_rgb)
            if m is not None and _area_ratio(m) >= MIN_RATIO:
                candidates.append(("shape", m))
        except Exception as e:
            log.debug(f"Fallback failed: {e}")
    if not candidates:
        raise RuntimeError("No valid tree mask found")
    name, best = max(candidates, key=lambda kv: _area_ratio(kv[1]))
    refined = refine_tree_mask(best, img_rgb)
    log.info(f"Segmentation used: {name}")
    return refined

def detect_tree_extremes_pca(mask: np.ndarray) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    pts = np.column_stack(np.where(mask > 0))
    if len(pts) == 0: return (0,0),(0,0)
    mu = pts.mean(axis=0, keepdims=True)
    X = pts - mu
    cov = (X.T @ X) / max(1, len(pts)-1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    proj = X @ v
    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))
    top = tuple(pts[i_min][::-1])
    base = tuple(pts[i_max][::-1])
    return base, top

def bilinear_sample(depth: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    H, W = depth.shape
    x0 = np.clip(np.floor(xs).astype(int), 0, W-1)
    x1 = np.clip(x0+1, 0, W-1)
    y0 = np.clip(np.floor(ys).astype(int), 0, H-1)
    y1 = np.clip(y0+1, 0, H-1)
    dx = xs - x0
    dy = ys - y0
    Ia = depth[y0, x0]
    Ib = depth[y0, x1]
    Ic = depth[y1, x0]
    Id = depth[y1, x1]
    top = Ia*(1-dx) + Ib*dx
    bot = Ic*(1-dx) + Id*dx
    return top*(1-dy) + bot*dy

def refine_extremes_with_depth(mask: np.ndarray, base: Tuple[int,int], top: Tuple[int,int], depth: np.ndarray) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    x_top, y_top = top; x_base, y_base = base
    N = 400
    xs = np.linspace(x_top, x_base, N)
    ys = np.linspace(y_top, y_base, N)
    d = bilinear_sample(depth, xs, ys)
    g = np.gradient(d)
    idx_top = np.argmax(np.abs(g[: max(20, int(0.4*N)) ]))
    idx_base_region_start = int(0.75*N)
    g_tail = np.abs(g[idx_base_region_start:])
    if len(g_tail) == 0:
        idx_base = N-1
    else:
        idx_base_local = np.argmin(g_tail)
        idx_base = idx_base_region_start + idx_base_local
    yH = mask.shape[0]
    candidates = np.where(ys >= 0.85*yH)[0]
    if len(candidates) > 0:
        g_c = np.abs(g[candidates])
        idx_base = candidates[np.argmin(g_c)]
    top_ref = (int(xs[idx_top]), int(ys[idx_top]))
    base_ref = (int(xs[idx_base]), int(ys[idx_base]))
    return base_ref, top_ref

def classify_tree_size(pixel_height: int) -> str:
    if pixel_height < 600: return "PETIT"
    if pixel_height < 1400: return "MOYEN"
    return "GRAND"

REF = {
    "PETIT": {"ref_px": 2185, "ref_m": 1.35, "factor": 0.90},
    "MOYEN": {"ref_px": 2185, "ref_m": 1.35, "factor": 1.00},
    "GRAND": {"ref_px": 2185, "ref_m": 1.35, "factor": 1.10},
}
UNC_PCT = {"PETIT": 0.15, "MOYEN": 0.25, "GRAND": 0.35}
DBH_RATIO = {"PETIT": 0.055, "MOYEN": 0.065, "GRAND": 0.075}

def estimate_height_and_dbh(pixel_height: int, manual_ppm: Optional[float] = None):
    size = classify_tree_size(pixel_height)
    if manual_ppm and manual_ppm > 0:
        height_m = pixel_height / manual_ppm
    else:
        ref = REF[size]
        height_m = (pixel_height / ref["ref_px"]) * ref["ref_m"] * ref["factor"]
    ci = max(0.02, height_m * UNC_PCT[size])
    dbh_m = height_m * DBH_RATIO[size]
    return height_m, ci, size, dbh_m

def render_overlay(img_bgr: np.ndarray, base: Tuple[int,int], top: Tuple[int,int], height_m: float, height_ci: float, size_label: str, dbh_m: float) -> np.ndarray:
    out = img_bgr.copy()
    cv2.circle(out, base, 6, (0, 255, 255), -1)
    cv2.circle(out, top,  6, (0, 0, 255), -1)
    cv2.line(out, base, top, (0, 255, 0), 2)
    txt = f"H={height_m:.2f}m ±{height_ci:.2f} | {size_label} | DBH≈{dbh_m:.2f}m"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (10, 10), (10 + tw + 10, 10 + th + 14), (0, 0, 0), -1)
    cv2.putText(out, txt, (16, 10 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def _quality_metrics(mask: np.ndarray, base: Tuple[int,int], top: Tuple[int,int]) -> dict:
    mr = _area_ratio(mask)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return {"mask_ratio": 0.0, "aspect_v": 0.0, "axis_len_px": 0}
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    axis_len = int(np.hypot(top[0]-base[0], top[1]-base[1]))
    return {"mask_ratio": round(mr, 4), "aspect_v": round(h/max(1,w), 3), "axis_len_px": axis_len}

def render_and_save_outputs(img_bgr: np.ndarray, img_rgb: np.ndarray, mask: np.ndarray, base: Tuple[int,int], top: Tuple[int,int], height_m: float, height_ci: float, size_label: str, dbh_m: float, save_debug: bool = False, source_path: Optional[str] = None) -> str:
    stem = "tree"
    if source_path: stem = os.path.splitext(os.path.basename(source_path))[0]
    out_img = render_overlay(img_bgr, base, top, height_m, height_ci, size_label, dbh_m)
    out_path = f"{stem}_annotated.jpg"
    cv2.imwrite(out_path, out_img)
    if save_debug:
        mask_path = f"{stem}_mask.png"
        json_path = f"{stem}_result.json"
        cv2.imwrite(mask_path, (mask > 0).astype("uint8") * 255)
        qm = _quality_metrics(mask, base, top)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "source": source_path,
                "height_m": round(height_m, 3),
                "uncertainty_m": round(height_ci, 3),
                "dbh_m": round(dbh_m, 3),
                "size_label": size_label,
                "base_xy": base,
                "top_xy": top,
                "image_shape_hw": img_rgb.shape[:2],
                "quality": qm
            }, ensure_ascii=False, indent=2)
        log.info(f"Saved debug: {mask_path}, {json_path}")
    return out_path

def load_midas(model_type: str = "DPT_Large", device="cpu"):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if "DPT" in model_type:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return midas, transform

def infer_depth(img_rgb: np.ndarray, model, transform, device="cpu") -> np.ndarray:
    import PIL.Image as Image
    inp = transform(Image.fromarray(img_rgb)).to(device)
    with torch.inference_mode():
        pred = model(inp)
        if isinstance(pred, (list, tuple)): pred = pred[0]
        pred = F.interpolate(pred.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
    depth = pred.detach().cpu().numpy().astype(np.float32)
    dmin, dmax = np.percentile(depth, 1), np.percentile(depth, 99)
    depth = np.clip((depth - dmin) / max(1e-6, (dmax - dmin)), 0, 1)
    depth = cv2.bilateralFilter(depth, 9, 0.1, 5.0)
    return depth

def main():
    args = _parse_args()
    img_path = get_image_path(args.image)
    if not img_path:
        log.error("No image provided/selected.")
        return

    img_bgr, img_rgb, H, W = load_image(img_path, max_side=args.max_size)
    log.info(f"Image loaded: {img_path} ({W}x{H})")

    if TORCH_OK and TV_OK:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Torch available: using device {device}")
    else:
        device = "cpu"
        log.info("Torch/torchvision not fully available; using color/shape segmentation only.")

    mask = segment_tree_multi_approach(img_rgb, device)
    base, top = detect_tree_extremes_pca(mask)

    if args.use_depth and TORCH_OK:
        try:
            midas, tfm = load_midas(args.depth_model, device)
            depth = infer_depth(img_rgb, midas, tfm, device)
            base, top = refine_extremes_with_depth(mask, base, top, depth)
            log.info("Depth-guided refinement applied.")
        except Exception as e:
            log.warning(f"Depth model unavailable or failed ({e}); proceeding without depth.")

    pixel_height = int(np.hypot(top[0] - base[0], top[1] - base[1]))
    log.info(f"Axis length (px) = {pixel_height}")

    height_m, height_ci, size_label, dbh_m = estimate_height_and_dbh(pixel_height, manual_ppm=args.pixel_per_meter)
    log.info(f"Estimated height: {height_m:.2f} m (±{height_ci:.2f} m), class={size_label}, DBH≈{dbh_m:.2f} m")

    output_path = render_and_save_outputs(img_bgr, img_rgb, mask, base, top, height_m, height_ci, size_label, dbh_m, save_debug=args.save_debug, source_path=img_path)
    log.info(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
