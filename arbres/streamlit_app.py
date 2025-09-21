import streamlit as st
import numpy as np
import cv2
import json
import os
from typing import Tuple, Optional, List
import logging

# Check for required packages
try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed. Please install it with: pip install streamlit")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("streamlit_arbres")

# Try to import torch-related modules
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

# Common functions

def load_image(img_bytes: bytes, max_side: int = 1600):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image")
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
    if num_labels <= 1:
        return mask_bool
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
    if mask_bool.sum() == 0:
        return None
    return (mask_bool.astype(np.uint8) * 255)

_MASKRCNN = None
def _get_maskrcnn(device):
    global _MASKRCNN
    if _MASKRCNN is None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        _MASKRCNN = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
    return _MASKRCNN

def try_maskrcnn_segmentation(img_rgb: np.ndarray, device="cpu") -> Optional[np.ndarray]:
    if not TORCH_OK or not TV_OK:
        return None
    model = _get_maskrcnn(device)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    inp = torch.from_numpy(x).unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode():
        out = model(inp)[0]
    masks = out.get("masks", None)
    if masks is None or len(masks) == 0:
        return None
    best, best_area = None, 0
    for m in masks.squeeze(1).detach().cpu().numpy():
        mb = (m > 0.5)
        area = mb.sum()
        if area > best_area:
            best, best_area = mb, area
    if best is None:
        return None
    return (best.astype(np.uint8) * 255)

def create_fallback_mask(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k, iterations=1)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k, iterations=1)
    mask_bin = _largest_component(mask_bin > 0).astype(np.uint8)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k2, iterations=1)
    return (mask_bin * 255)

def _seg_color_at_scale(img_rgb: np.ndarray, scale: float) -> Optional[np.ndarray]:
    if abs(scale - 1.0) < 1e-3:
        return create_vegetation_mask_by_color(img_rgb)
    h, w = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    m = create_vegetation_mask_by_color(small)
    if m is None:
        return None
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

def _tta_union_color(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    masks: List[np.ndarray] = []
    for s in (0.75, 1.0, 1.25):
        m = _seg_color_at_scale(img_rgb, s)
        if m is not None:
            masks.append(m > 0)
    if not masks:
        return None
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
    if len(pts) == 0:
        return (0,0),(0,0)
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

_MIDAS = None
_MIDAS_TRANSFORM = None
def load_midas(model_type: str = "DPT_Large", device="cpu"):
    global _MIDAS, _MIDAS_TRANSFORM
    if _MIDAS is None:
        _MIDAS = torch.hub.load("intel-isl/MiDaS", model_type)
        _MIDAS.to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "DPT" in model_type:
            _MIDAS_TRANSFORM = transforms.dpt_transform
        else:
            _MIDAS_TRANSFORM = transforms.small_transform
    return _MIDAS, _MIDAS_TRANSFORM

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
    cv2.putText(out, txt, (16, 10 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
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

# Model 1: Fast Color+Axis Estimator
def run_fast_color_estimator(img_rgb: np.ndarray, device="cpu", ppm=None):
    mask = segment_tree_multi_approach(img_rgb, device)
    base, top = detect_tree_extremes_pca(mask)
    pixel_height = int(np.hypot(top[0] - base[0], top[1] - base[1]))
    height_m, ci, size_label, dbh_m = estimate_height_and_dbh(pixel_height, ppm)
    return mask, base, top, height_m, ci, size_label, dbh_m, pixel_height

# Model 2: Depth-Guided Axis Estimator
def run_depth_guided_estimator(img_rgb: np.ndarray, device="cpu", ppm=None, use_depth=True):
    mask = segment_tree_multi_approach(img_rgb, device)
    base, top = detect_tree_extremes_pca(mask)
    if use_depth and TORCH_OK:
        try:
            midas, tfm = load_midas("DPT_Large", device)
            depth = infer_depth(img_rgb, midas, tfm, device)
            base, top = refine_extremes_with_depth(mask, base, top, depth)
        except Exception as e:
            st.warning(f"Depth refinement failed: {e}")
    pixel_height = int(np.hypot(top[0] - base[0], top[1] - base[1]))
    height_m, ci, size_label, dbh_m = estimate_height_and_dbh(pixel_height, ppm)
    return mask, base, top, height_m, ci, size_label, dbh_m, pixel_height

# Model 3: Arbres QC Console (adapted for batch processing)
def run_qc_console_estimator(img_rgb: np.ndarray, device="cpu", ppm=None):
    # Similar to fast color, but could add batch processing
    mask = segment_tree_multi_approach(img_rgb, device)
    base, top = detect_tree_extremes_pca(mask)
    pixel_height = int(np.hypot(top[0] - base[0], top[1] - base[1]))
    height_m, ci, size_label, dbh_m = estimate_height_and_dbh(pixel_height, ppm)
    return mask, base, top, height_m, ci, size_label, dbh_m, pixel_height

# Streamlit UI

st.title("Arbres Estimators - Streamlit App")

tab1, tab2, tab3 = st.tabs(["Fast Color Estimator", "Depth-Guided Estimator", "QC Console Estimator"])

with tab1:
    st.header("Fast Color+Axis Estimator")
    uploaded_files1 = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True, key="uploader1")
    max_size1 = st.slider("Max Image Size", 800, 2000, 1600, key="max_size1")
    ppm1 = st.number_input("Pixels per Meter (optional)", value=0.0, min_value=0.0, key="ppm1")
    save_debug1 = st.checkbox("Save Debug Info", key="debug1")
    
    if uploaded_files1 and st.button("Process", key="process1"):
        for uploaded_file in uploaded_files1:
            img_bytes = uploaded_file.read()
            try:
                img_bgr, img_rgb, H, W = load_image(img_bytes, max_size1)
                device = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"
                mask, base, top, height_m, ci, size_label, dbh_m, pixel_height = run_fast_color_estimator(img_rgb, device, ppm1 if ppm1 > 0 else None)
                annotated = render_overlay(img_bgr, base, top, height_m, ci, size_label, dbh_m)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption=f"Original: {uploaded_file.name}", use_column_width=True)
                with col2:
                    st.image(annotated, caption="Annotated", use_column_width=True)
                st.write(f"**Height:** {height_m:.2f} m ± {ci:.2f} m")
                st.write(f"**Size Class:** {size_label}")
                st.write(f"**DBH:** {dbh_m:.2f} m")
                st.write(f"**Pixel Height:** {pixel_height}")
                if save_debug1:
                    qm = _quality_metrics(mask, base, top)
                    st.json(qm)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

with tab2:
    st.header("Depth-Guided Axis Estimator")
    uploaded_files2 = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True, key="uploader2")
    max_size2 = st.slider("Max Image Size", 800, 2000, 1600, key="max_size2")
    ppm2 = st.number_input("Pixels per Meter (optional)", value=0.0, min_value=0.0, key="ppm2")
    use_depth2 = st.checkbox("Use Depth Guidance", value=True, key="use_depth2")
    save_debug2 = st.checkbox("Save Debug Info", key="debug2")
    
    if uploaded_files2 and st.button("Process", key="process2"):
        for uploaded_file in uploaded_files2:
            img_bytes = uploaded_file.read()
            try:
                img_bgr, img_rgb, H, W = load_image(img_bytes, max_size2)
                device = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"
                mask, base, top, height_m, ci, size_label, dbh_m, pixel_height = run_depth_guided_estimator(img_rgb, device, ppm2 if ppm2 > 0 else None, use_depth2)
                annotated = render_overlay(img_bgr, base, top, height_m, ci, size_label, dbh_m)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption=f"Original: {uploaded_file.name}", use_column_width=True)
                with col2:
                    st.image(annotated, caption="Annotated", use_column_width=True)
                st.write(f"**Height:** {height_m:.2f} m ± {ci:.2f} m")
                st.write(f"**Size Class:** {size_label}")
                st.write(f"**DBH:** {dbh_m:.2f} m")
                st.write(f"**Pixel Height:** {pixel_height}")
                if save_debug2:
                    qm = _quality_metrics(mask, base, top)
                    st.json(qm)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

with tab3:
    st.header("QC Console Estimator")
    uploaded_files3 = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True, key="uploader3")
    max_size3 = st.slider("Max Image Size", 800, 2000, 1600, key="max_size3")
    ppm3 = st.number_input("Pixels per Meter (optional)", value=0.0, min_value=0.0, key="ppm3")
    save_debug3 = st.checkbox("Save Debug Info", key="debug3")
    
    if uploaded_files3 and st.button("Process", key="process3"):
        for uploaded_file in uploaded_files3:
            img_bytes = uploaded_file.read()
            try:
                img_bgr, img_rgb, H, W = load_image(img_bytes, max_size3)
                device = "cuda" if TORCH_OK and torch.cuda.is_available() else "cpu"
                mask, base, top, height_m, ci, size_label, dbh_m, pixel_height = run_qc_console_estimator(img_rgb, device, ppm3 if ppm3 > 0 else None)
                annotated = render_overlay(img_bgr, base, top, height_m, ci, size_label, dbh_m)
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption=f"Original: {uploaded_file.name}", use_column_width=True)
                with col2:
                    st.image(annotated, caption="Annotated", use_column_width=True)
                st.write(f"**Height:** {height_m:.2f} m ± {ci:.2f} m")
                st.write(f"**Size Class:** {size_label}")
                st.write(f"**DBH:** {dbh_m:.2f} m")
                st.write(f"**Pixel Height:** {pixel_height}")
                if save_debug3:
                    qm = _quality_metrics(mask, base, top)
                    st.json(qm)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
