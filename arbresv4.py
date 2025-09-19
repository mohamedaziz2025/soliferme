from __future__ import annotations
import argparse
import logging
import os
import json
from typing import Tuple, Optional
import numpy as np
import cv2

try:
    import torch
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    )
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    torch = None

try:
    import tkinter as tk
    from tkinter import filedialog
    TK_OK = True
except Exception:
    TK_OK = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("arbres")

def _parse_args():
    p = argparse.ArgumentParser(description="Tree height estimator")
    p.add_argument("--image", type=str, help="Path to input image")
    p.add_argument("--save-debug", action="store_true")
    p.add_argument("--max-size", type=int, default=1600)
    p.add_argument("--pixel-per-meter", type=float, default=None)
    return p.parse_args()

def select_image_file() -> Optional[str]:
    if not TK_OK:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
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
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    return img_bgr, img_rgb, H, W

def _area_ratio(mask: np.ndarray) -> float:
    if mask.dtype != np.bool_:
        mask_bin = mask > 0
    else:
        mask_bin = mask
    return float(mask_bin.sum()) / (mask_bin.shape[0] * mask_bin.shape[1])

def _largest_component(mask_bool: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask_bool
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)
    return labels == idx

def create_vegetation_mask_by_color(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([20, 30, 30], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask_bool = mask > 0
    mask_bool = _largest_component(mask_bool)
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    return (mask_bool.astype(np.uint8) * 255)

_MASKRCNN = None
def _get_maskrcnn(device):
    global _MASKRCNN
    if _MASKRCNN is None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        _MASKRCNN = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
    return _MASKRCNN

def try_maskrcnn_segmentation(img_rgb: np.ndarray, device="cpu") -> Optional[np.ndarray]:
    if not TORCH_OK:
        return None
    model = _get_maskrcnn(device)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    inp = torch.from_numpy(x).unsqueeze(0)
    inp = inp.to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model(inp)[0]
    masks = out.get("masks", None)
    if masks is None or len(masks) == 0:
        return None
    best = None
    best_area = 0
    for m in masks.squeeze(1).detach().cpu().numpy():
        mb = (m > 0.5)
        area = mb.sum()
        if area > best_area:
            best = mb
            best_area = area
    if best is None:
        return None
    return (best.astype(np.uint8) * 255)

def create_fallback_mask(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    H, W = gray.shape
    MIN_BLOB_RATIO = 0.02
    mask = np.zeros_like(gray, dtype=np.uint8)
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= MIN_BLOB_RATIO * H * W:
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    if mask.sum() == 0:
        return None
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

def segment_tree_multi_approach(img_rgb: np.ndarray, device="cpu") -> np.ndarray:
    H, W, _ = img_rgb.shape
    MIN_RATIO = 0.005
    candidates: list[tuple[str, np.ndarray]] = []
    try:
        m = create_vegetation_mask_by_color(img_rgb)
        if m is not None and _area_ratio(m) >= MIN_RATIO:
            candidates.append(("color", m))
    except Exception as e:
        log.debug(f"Color segmentation failed: {e}")
    if TORCH_OK:
        try:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = try_maskrcnn_segmentation(img_rgb, device=dev)
            if m is not None and _area_ratio(m) >= MIN_RATIO:
                candidates.append(("rcnn", m))
        except Exception as e:
            log.debug(f"Mask R-CNN segmentation failed: {e}")
    if not candidates:
        try:
            m = create_fallback_mask(img_rgb)
            if m is not None and _area_ratio(m) >= MIN_RATIO:
                candidates.append(("shape", m))
        except Exception as e:
            log.debug(f"Fallback segmentation failed: {e}")
    if not candidates:
        raise RuntimeError("No valid tree mask found")
    name, best = max(candidates, key=lambda kv: _area_ratio(kv[1]))
    refined = refine_tree_mask(best, img_rgb)
    log.info(f"Segmentation approach used: {name}")
    return refined

def detect_tree_extremes(mask: np.ndarray, erode_ksize: int = 3) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    mb = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    clean = cv2.erode(mb, k, iterations=1)
    ys, xs = np.where(clean > 0)
    if len(ys) == 0:
        ys, xs = np.where(mb > 0)
        if len(ys) == 0:
            return (0, 0), (0, 0)
    top_y = int(ys.min())
    base_y = int(ys.max())
    def robust_x_at(y: int, win: int = 5) -> int:
        y0, y1 = max(0, y - win), min(mb.shape[0], y + win + 1)
        col_idx = np.where(np.any(clean[y0:y1] > 0, axis=0))[0]
        if len(col_idx) == 0:
            col_idx = xs
        return int(np.median(col_idx))
    top_x = robust_x_at(top_y)
    base_x = robust_x_at(base_y)
    return (base_x, base_y), (top_x, top_y)

def classify_tree_size(pixel_height: int) -> str:
    if pixel_height < 600:
        return "PETIT"
    if pixel_height < 1400:
        return "MOYEN"
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

def render_overlay(img_bgr: np.ndarray,
                   base: Tuple[int,int],
                   top: Tuple[int,int],
                   height_m: float,
                   height_ci: float,
                   size_label: str,
                   dbh_m: float) -> np.ndarray:
    out = img_bgr.copy()
    cv2.circle(out, base, 6, (0, 255, 255), -1)
    cv2.circle(out, top,  6, (0, 0, 255), -1)
    cv2.line(out, base, top, (0, 255, 0), 2)
    txt = f"H={height_m:.2f}m ±{height_ci:.2f}  |  {size_label}  |  DBH≈{dbh_m:.2f}m"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (10, 10), (10 + tw + 10, 10 + th + 14), (0, 0, 0), -1)
    cv2.putText(out, txt, (16, 10 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def render_and_save_outputs(img_bgr: np.ndarray,
                            img_rgb: np.ndarray,
                            mask: np.ndarray,
                            base: Tuple[int,int],
                            top: Tuple[int,int],
                            height_m: float,
                            height_ci: float,
                            size_label: str,
                            dbh_m: float,
                            save_debug: bool = False,
                            source_path: Optional[str] = None) -> str:
    stem = "tree"
    if source_path:
        stem = os.path.splitext(os.path.basename(source_path))[0]
    out_img = render_overlay(img_bgr, base, top, height_m, height_ci, size_label, dbh_m)
    out_path = f"{stem}_annotated.jpg"
    cv2.imwrite(out_path, out_img)
    if save_debug:
        mask_path = f"{stem}_mask.png"
        json_path = f"{stem}_result.json"
        cv2.imwrite(mask_path, (mask > 0).astype("uint8") * 255)
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
            }, ensure_ascii=False, indent=2)
        log.info(f"Saved debug: {mask_path}, {json_path}")
    return out_path

def main():
    args = _parse_args()
    img_path = get_image_path(args.image)
    if not img_path:
        log.error("No image provided/selected.")
        return
    if TORCH_OK:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Torch available: using device {device}")
    else:
        device = "cpu"
        log.info("Torch/torchvision not available; using color/shape segmentation only.")
    img_bgr, img_rgb, H, W = load_image(img_path, max_side=args.max_size)
    log.info(f"Image loaded: {img_path} ({W}x{H})")
    tree_mask = segment_tree_multi_approach(img_rgb, device)
    (base_x, base_y), (top_x, top_y) = detect_tree_extremes(tree_mask)
    pixel_height = abs(top_y - base_y)
    log.info(f"Pixel height = {pixel_height}")
    height_m, height_ci, size_label, dbh_m = estimate_height_and_dbh(
        pixel_height, manual_ppm=args.pixel_per_meter
    )
    log.info(f"Estimated height: {height_m:.2f} m (±{height_ci:.2f} m), class={size_label}, DBH≈{dbh_m:.2f} m")
    output_path = render_and_save_outputs(
        img_bgr, img_rgb, tree_mask, (base_x, base_y), (top_x, top_y),
        height_m, height_ci, size_label, dbh_m,
        save_debug=args.save_debug, source_path=img_path
    )
    log.info(f"Saved: {output_path}")

if __name__ == "__main__":
    main()