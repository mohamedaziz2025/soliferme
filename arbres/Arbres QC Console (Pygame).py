from __future__ import annotations
import argparse, logging, os, json, sys, glob
from typing import Tuple, Optional, List
import numpy as np
import cv2
import pygame

try:
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    torch = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("arbres_pygame")

# --------------------------- Core estimation pipeline ---------------------------

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

def load_image(path: str, max_side: int = 1600):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    return img_bgr, img_rgb, H, W

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
    if not TORCH_OK:
        return None
    model = _get_maskrcnn(device)
    x = img_rgb.astype(np.float32)/255.0
    x = np.transpose(x, (2,0,1))
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
    if TORCH_OK:
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

def _quality_metrics(mask: np.ndarray, base: Tuple[int,int], top: Tuple[int,int]) -> dict:
    mr = _area_ratio(mask)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return {"mask_ratio": 0.0, "aspect_v": 0.0, "axis_len_px": 0}
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    axis_len = int(np.hypot(top[0]-base[0], top[1]-base[1]))
    return {"mask_ratio": round(mr, 4), "aspect_v": round(h/max(1,w), 3), "axis_len_px": axis_len}

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

# --------------------------- Pygame UI helpers ---------------------------

def np_to_surface(img_rgb: np.ndarray) -> pygame.Surface:
    return pygame.image.frombuffer(img_rgb.tobytes(), img_rgb.shape[1::-1], "RGB")

def draw_mask_overlay(surface: pygame.Surface, mask: np.ndarray, color=(0,255,0), alpha=90):
    h, w = mask.shape
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    arr = pygame.surfarray.pixels_alpha(overlay)
    arr[:] = 0
    del arr
    mask_inds = np.where(mask > 0)
    for y, x in zip(mask_inds[0], mask_inds[1]):
        overlay.set_at((x, y), (*color, alpha))
    surface.blit(overlay, (0, 0))

def scale_for_window(img_rgb: np.ndarray, max_w=1280, max_h=800):
    H, W = img_rgb.shape[:2]
    s = min(max_w / W, max_h / H, 1.0)
    newW, newH = int(W*s), int(H*s)
    return s, newW, newH

def blit_centered(screen, surf):
    sw, sh = surf.get_size()
    screen.blit(surf, (0,0))
    return (0,0)

# --------------------------- Session model ---------------------------

class Session:
    def __init__(self, paths: List[str], save_dir: str, max_side: int, ppm: Optional[float], save_debug: bool):
        self.paths = paths
        self.i = 0
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_side = max_side
        self.ppm = ppm
        self.save_debug = save_debug
        self.cache = {}

    def current_path(self) -> str:
        return self.paths[self.i]

    def load_and_process(self, path: str):
        if path in self.cache:
            return self.cache[path]
        img_bgr, img_rgb, H, W = load_image(path, self.max_side)
        device = torch.device("cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu") if TORCH_OK else "cpu"
        mask = segment_tree_multi_approach(img_rgb, device)
        (bx, by), (tx, ty) = detect_tree_extremes_pca(mask)
        pix_h = int(np.hypot(tx - bx, ty - by))
        h_m, ci, lab, dbh = estimate_height_and_dbh(pix_h, manual_ppm=self.ppm)
        res = {
            "img_bgr": img_bgr, "img_rgb": img_rgb, "mask": mask.copy(),
            "base": [bx, by], "top": [tx, ty],
            "pix_h": pix_h, "h_m": h_m, "ci": ci, "label": lab, "dbh": dbh
        }
        self.cache[path] = res
        return res

    def save_outputs(self, path: str, state: dict):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_img = render_overlay(state["img_bgr"], tuple(state["base"]), tuple(state["top"]), state["h_m"], state["ci"], state["label"], state["dbh"])
        cv2.imwrite(os.path.join(self.save_dir, f"{stem}_annotated.jpg"), out_img)
        if self.save_debug:
            cv2.imwrite(os.path.join(self.save_dir, f"{stem}_mask.png"), (state["mask"]>0).astype("uint8")*255)
            qm = _quality_metrics(state["mask"], tuple(state["base"]), tuple(state["top"]))
            with open(os.path.join(self.save_dir, f"{stem}_result.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "source": path,
                    "height_m": round(state["h_m"],3),
                    "uncertainty_m": round(state["ci"],3),
                    "dbh_m": round(state["dbh"],3),
                    "size_label": state["label"],
                    "base_xy": state["base"],
                    "top_xy": state["top"],
                    "image_shape_hw": state["img_rgb"].shape[:2],
                    "quality": qm
                }, ensure_ascii=False, indent=2)

    def recompute_from_mask(self, state: dict):
        (bx, by), (tx, ty) = detect_tree_extremes_pca(state["mask"])
        state["base"] = [bx, by]
        state["top"]  = [tx, ty]
        state["pix_h"] = int(np.hypot(tx - bx, ty - by))
        h_m, ci, lab, dbh = estimate_height_and_dbh(state["pix_h"], manual_ppm=self.ppm)
        state["h_m"], state["ci"], state["label"], state["dbh"] = h_m, ci, lab, dbh

# --------------------------- Pygame app ---------------------------

def run_ui(input_paths: List[str], save_dir: str, max_side: int, ppm: Optional[float], save_debug: bool):
    pygame.init()
    screen = pygame.display.set_mode((1280, 800))
    pygame.display.set_caption("Arbres Reviewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    sess = Session(input_paths, save_dir, max_side, ppm, save_debug)

    brush_radius = 12
    tool = "none"  # 'add', 'erase', 'move_base', 'move_top'
    dragging = False

    def render_state(state):
        s, newW, newH = scale_for_window(state["img_rgb"], 1280, 800)
        rgb_scaled = cv2.resize(state["img_rgb"], (newW, newH), interpolation=cv2.INTER_AREA)
        mask_scaled = cv2.resize(state["mask"], (newW, newH), interpolation=cv2.INTER_NEAREST)
        base_s = (int(state["base"][0]*s), int(state["base"][1]*s))
        top_s  = (int(state["top"][0]*s), int(state["top"][1]*s))
        surf = np_to_surface(rgb_scaled.copy())
        draw_mask_overlay(surf, mask_scaled, color=(0,255,0), alpha=80)
        pygame.draw.circle(surf, (255,255,0), base_s, 6)
        pygame.draw.circle(surf, (255,0,0), top_s, 6)
        pygame.draw.line(surf, (0,255,0), base_s, top_s, 2)
        blit_centered(screen, surf)
        hud = f"[{sess.i+1}/{len(sess.paths)}] {os.path.basename(sess.current_path())} | H={state['h_m']:.2f}m ±{state['ci']:.2f} | DBH≈{state['dbh']:.2f}m | Tool:{tool} | Brush:{brush_radius}px"
        txt = font.render(hud, True, (255,255,255))
        screen.blit(txt, (10, 10))

    def edit_mask_at(state, mx, my, mode):
        s, newW, newH = scale_for_window(state["img_rgb"], 1280, 800)
        x = int(mx / s); y = int(my / s)
        h, w = state["mask"].shape
        if not (0 <= x < w and 0 <= y < h): return
        rr = brush_radius
        y0,y1 = max(0,y-rr), min(h, y+rr+1)
        x0,x1 = max(0,x-rr), min(w, x+rr+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        m = (xx - x)**2 + (yy - y)**2 <= rr*rr
        if mode == "add":
            state["mask"][y0:y1, x0:x1][m] = 255
        elif mode == "erase":
            state["mask"][y0:y1, x0:x1][m] = 0
        sess.recompute_from_mask(state)

    def move_point(state, mx, my, which):
        s, _, _ = scale_for_window(state["img_rgb"], 1280, 800)
        x = int(mx / s); y = int(my / s)
        if which == "base": state["base"] = [x, y]
        if which == "top":  state["top"]  = [x, y]
        state["pix_h"] = int(np.hypot(state["top"][0]-state["base"][0], state["top"][1]-state["base"][1]))
        h_m, ci, lab, dbh = estimate_height_and_dbh(state["pix_h"], manual_ppm=sess.ppm)
        state["h_m"], state["ci"], state["label"], state["dbh"] = h_m, ci, lab, dbh

    running = True
    state = sess.load_and_process(sess.current_path())

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: running = False
                elif event.key == pygame.K_n:
                    sess.i = min(sess.i+1, len(sess.paths)-1)
                    state = sess.load_and_process(sess.current_path())
                elif event.key == pygame.K_p:
                    sess.i = max(sess.i-1, 0)
                    state = sess.load_and_process(sess.current_path())
                elif event.key == pygame.K_s:
                    sess.save_outputs(sess.current_path(), state)
                    log.info("Saved outputs")
                elif event.key == pygame.K_a: tool = "add"
                elif event.key == pygame.K_e: tool = "erase"
                elif event.key == pygame.K_b: tool = "move_base"
                elif event.key == pygame.K_t: tool = "move_top"
                elif event.key == pygame.K_SPACE: tool = "none"
                elif event.key == pygame.K_LEFTBRACKET: brush_radius = max(2, brush_radius-2)
                elif event.key == pygame.K_RIGHTBRACKET: brush_radius = min(64, brush_radius+2)
                elif event.key == pygame.K_r:
                    # recompute from current mask
                    sess.recompute_from_mask(state)
                elif event.key == pygame.K_m:
                    # re-run full segmentation (useful after big edits)
                    img_rgb = state["img_rgb"]
                    device = torch.device("cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu") if TORCH_OK else "cpu"
                    mask = segment_tree_multi_approach(img_rgb, device)
                    state["mask"] = mask
                    sess.recompute_from_mask(state)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 3):
                    dragging = True
                    mx, my = pygame.mouse.get_pos()
                    if tool in ("add","erase"):
                        edit_mask_at(state, mx, my, tool)
                    elif tool == "move_base":
                        move_point(state, mx, my, "base")
                    elif tool == "move_top":
                        move_point(state, mx, my, "top")

            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = pygame.mouse.get_pos()
                if tool in ("add","erase"):
                    edit_mask_at(state, mx, my, tool)
                elif tool == "move_base":
                    move_point(state, mx, my, "base")
                elif tool == "move_top":
                    move_point(state, mx, my, "top")

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False

        screen.fill((0,0,0))
        render_state(state)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# --------------------------- CLI ---------------------------

def collect_images(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.JPEG","*.PNG","*.BMP")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(input_path, e)))
        files.sort()
        return files
    else:
        return [input_path]

def main():
    ap = argparse.ArgumentParser(description="Arbres Pygame Reviewer")
    ap.add_argument("--input", required=True, help="Image file or directory")
    ap.add_argument("--save-dir", default="out", help="Where to write outputs")
    ap.add_argument("--max-size", type=int, default=1600)
    ap.add_argument("--pixel-per-meter", type=float, default=None)
    ap.add_argument("--save-debug", action="store_true")
    args = ap.parse_args()

    paths = collect_images(args.input)
    if not paths:
        print("No images found.")
        sys.exit(1)
    run_ui(paths, args.save_dir, args.max_size, args.pixel_per_meter, args.save_debug)

if __name__ == "__main__":
    main()
