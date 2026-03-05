"""
Detection script using GroundingDINO (open-vocabulary) + SAM (segmentation)
to detect dishes and decide if they're IN SINK based on *mask overlap* with a sink polygon.

RUN (use the project venv so dependencies are found):
  .venv/bin/python detection_dino_sam.py
  # or: source .venv/bin/activate  then  python detection_dino_sam.py

  python detection_dino_sam.py --videos clip1.mp4 clip2.mp4
  python detection_dino_sam.py --calibrate
  python detection_dino_sam.py --device mps   # Mac (Apple Silicon) acceleration if available

Dependencies (install in .venv):
  pip install opencv-python numpy torch torchvision aiohttp
  pip install git+https://github.com/IDEA-Research/GroundingDINO.git
  pip install git+https://github.com/facebookresearch/segment-anything.git

Weights (set paths below):
  - GroundingDINO config: GroundingDINO_SwinT_OGC.py  (in repo)
  - GroundingDINO ckpt:   groundingdino_swint_ogc.pth  (download, see error message if missing)
  - SAM ckpt:             sam_vit_b_01ec64.pth (or vit_l/vit_h)
"""

try:
    import cv2
except ModuleNotFoundError as e:
    if "cv2" in str(e):
        print("Missing opencv. Use the project venv and install dependencies:")
        print("  .venv/bin/python detection_dino_sam.py")
        print("  pip install opencv-python numpy torch torchvision")
    raise

import json
import numpy as np
import time
import os
import re
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple

import torch

# ---------------- Transformers compatibility for GroundingDINO ----------------
# Newer transformers: get_head_mask removed; get_extended_attention_mask(attention_mask, input_shape, dtype=None)
# but GroundingDINO calls (attention_mask, input_shape, device), so 3rd arg becomes dtype=torch.device -> TypeError.
def _patch_bert_for_groundingdino():
    from transformers import BertModel
    # 1) Restore get_head_mask if missing (old API)
    if getattr(BertModel, "get_head_mask", None) is None:
        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
            head_mask = head_mask.to(dtype=head_mask.dtype)
            return head_mask

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                head_mask = _convert_head_mask_to_5d(self, head_mask, num_hidden_layers)
                if is_attention_chunked is True:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        BertModel._convert_head_mask_to_5d = _convert_head_mask_to_5d
        BertModel.get_head_mask = get_head_mask

    # 2) Wrap get_extended_attention_mask so (attention_mask, input_shape, device) -> (attention_mask, input_shape)
    _orig_get_extended = getattr(BertModel, "get_extended_attention_mask", None)
    if _orig_get_extended is not None and not getattr(_orig_get_extended, "_gdino_patched", False):
        def get_extended_attention_mask_compat(self, attention_mask, input_shape, third=None):
            # GroundingDINO passes device as 3rd arg; newer transformers expect dtype. Ignore device.
            if isinstance(third, torch.device):
                return _orig_get_extended(self, attention_mask, input_shape, dtype=None)
            return _orig_get_extended(self, attention_mask, input_shape, dtype=third)
        get_extended_attention_mask_compat._gdino_patched = True
        BertModel.get_extended_attention_mask = get_extended_attention_mask_compat

_patch_bert_for_groundingdino()

# ---------------- CONFIG ----------------
# Use generic nouns (open-vocab works better than overly specific prompts)
PROMPT_TEXT = "cup. mug. bowl. plate. pan. pot. glass. fork. spoon. knife. utensil."
DISH_VIDEOS_DIR = "./dish_videos"
SINK_POLYGON_FILE = "sink_polygon.json"
DEBUG = False

# DINO thresholds
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
NMS_IOU = 0.55

# Run detection every N frames (speed)
DETECT_EVERY = 3

# Decide "in sink" if (mask ∩ sink) / mask >= this
MASK_OVERLAP_THRESHOLD = 0.15

# Crop padding around sink polygon bounding rect (like your YOLO script)
CROP_PAD = 60

# Display sizing (like your UI)
FRAME_SIZE = (800, 600)
VIEW_W, VIEW_H = 650, 600

# Weights paths (EDIT THESE)
DINO_CONFIG_PATH = "./weights/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT_PATH = "./weights/groundingdino_swint_ogc.pth"
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT_PATH = "./weights/sam_vit_b_01ec64.pth"

# Blink filenames look like: mini-2-0eew-2026-02-19t01-21-26-00-00.mp4
BLINK_FNAME_PATTERN = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})t(\d{2})-(\d{2})-(\d{2})", re.IGNORECASE
)

# -------------- Helpers: video listing --------------
def list_videos_in_dish_videos():
    if not os.path.isdir(DISH_VIDEOS_DIR):
        return []
    return sorted(f for f in os.listdir(DISH_VIDEOS_DIR) if f.lower().endswith(".mp4"))

def get_video_paths_by_names(names):
    if not os.path.isdir(DISH_VIDEOS_DIR):
        return [], list(names)
    available = set(list_videos_in_dish_videos())
    resolved, missing = [], []
    for n in names:
        n = n.strip()
        if not n:
            continue
        base = os.path.basename(n)
        if base in available:
            resolved.append(os.path.join(DISH_VIDEOS_DIR, base))
        elif n in available:
            resolved.append(os.path.join(DISH_VIDEOS_DIR, n))
        else:
            missing.append(n if n != base else base)
    return resolved, missing

def get_videos_from_dish_videos(since_dt=None):
    if not os.path.isdir(DISH_VIDEOS_DIR):
        return []
    files = []
    for f in os.listdir(DISH_VIDEOS_DIR):
        if not f.lower().endswith(".mp4"):
            continue
        path = os.path.join(DISH_VIDEOS_DIR, f)
        if since_dt is not None:
            mo = BLINK_FNAME_PATTERN.search(f)
            if mo:
                y, m, d, h, mi, s = map(int, mo.groups())
                file_dt = datetime(y, m, d, h, mi, s)
            else:
                file_dt = datetime.fromtimestamp(os.path.getmtime(path))
            if file_dt < since_dt:
                continue
        files.append(path)
    return sorted(files)

# -------------- Sink polygon --------------
def get_sink_polygon():
    if os.path.isfile(SINK_POLYGON_FILE):
        with open(SINK_POLYGON_FILE) as f:
            points = json.load(f)
        return np.array(points, np.int32)
    # fallback
    return np.array([[150, 100], [500, 100], [500, 380], [150, 380]], np.int32)

def polygon_to_mask(poly: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    return m.astype(bool)

# -------------- NMS --------------
def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1 + 1)
        inter_h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = inter_w * inter_h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def clamp_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes

def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = xywh.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

# -------------- Models --------------
@dataclass
class Models:
    dino: object
    sam_predictor: object
    device: str

def choose_device(user_device: str = "") -> str:
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_models(device: str) -> Models:
    from groundingdino.util.inference import load_model as dino_load_model
    from segment_anything import sam_model_registry, SamPredictor

    if not os.path.isfile(DINO_CONFIG_PATH):
        raise FileNotFoundError(f"Missing DINO config: {DINO_CONFIG_PATH}")
    if not os.path.isfile(DINO_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Missing DINO checkpoint: {DINO_CHECKPOINT_PATH}\n"
            "Download it and put in ./weights/:\n"
            "  https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth\n"
            "  or from GitHub: https://github.com/IDEA-Research/GroundingDINO#install"
        )
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Missing SAM checkpoint: {SAM_CHECKPOINT_PATH}")

    dino = dino_load_model(DINO_CONFIG_PATH, DINO_CHECKPOINT_PATH, device=device)
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return Models(dino=dino, sam_predictor=predictor, device=device)

def _dino_preprocess_image(frame_bgr: np.ndarray):
    """Preprocess BGR image for GroundingDINO: same transform as official load_image (resize, normalize)."""
    from PIL import Image
    import groundingdino.datasets.transforms as T

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image_tensor, _ = transform(pil_image, None)
    return image_tensor


def dino_predict_boxes(dino_model, frame_bgr: np.ndarray, prompt: str, box_thresh: float, text_thresh: float, device: str):
    """
    Returns boxes_xyxy in pixel coords, scores, phrases.
    Handles both normalized cxcywh and pixel xyxy outputs (heuristic).
    """
    from groundingdino.util.inference import predict as dino_predict

    image_tensor = _dino_preprocess_image(frame_bgr)
    boxes, logits, phrases = dino_predict(
        model=dino_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        device=device,
    )

    boxes = boxes if isinstance(boxes, np.ndarray) else boxes.cpu().numpy()
    logits = logits if isinstance(logits, np.ndarray) else logits.cpu().numpy()
    phrases = list(phrases)

    if boxes.shape[0] > 0:
        # If values look normalized (<=1.5), treat as cxcywh normalized
        if float(np.max(boxes)) <= 1.5:
            h, w = frame_bgr.shape[:2]
            xyxy = xywh_to_xyxy(boxes)
            xyxy[:, [0, 2]] *= w
            xyxy[:, [1, 3]] *= h
            boxes = xyxy

    return boxes.astype(np.float32), logits.astype(np.float32), phrases

def sam_segment(sam_predictor, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> List[np.ndarray]:
    from segment_anything import SamPredictor  # noqa
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(rgb)

    masks = []
    for box in boxes_xyxy:
        box_np = box.astype(np.float32)
        m, scores, _ = sam_predictor.predict(box=box_np, multimask_output=True)
        best = int(np.argmax(scores))
        masks.append(m[best].astype(bool))
    return masks

def overlap_fraction(mask: np.ndarray, sink_mask: np.ndarray) -> float:
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    inter = float(np.logical_and(mask, sink_mask).sum())
    return inter / area

# -------------- Main processing --------------
def process_video(video_path: str, models: Models):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    original_sink_poly = get_sink_polygon()

    # consistent base coords
    base_w, base_h = FRAME_SIZE

    # crop around sink
    sx, sy, sw, sh = cv2.boundingRect(original_sink_poly)
    pad = CROP_PAD
    x1, y1 = max(0, sx - pad), max(0, sy - pad)
    x2, y2 = min(base_w, sx + sw + pad), min(base_h, sy + sh + pad)

    sink_poly_cropped = original_sink_poly - [x1, y1]

    # display scaling
    scale_x = VIEW_W / (x2 - x1)
    scale_y = VIEW_H / (y2 - y1)
    sink_poly_scaled = (sink_poly_cropped * [scale_x, scale_y]).astype(np.int32)

    # keep last detections for skipped frames
    last = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame = cv2.resize(frame, FRAME_SIZE)

        crop = frame[y1:y2, x1:x2].copy()
        crop_h, crop_w = crop.shape[:2]

        do_detect = (frame_idx % max(1, DETECT_EVERY) == 0)

        if do_detect:
            boxes, scores, phrases = dino_predict_boxes(
                models.dino, crop, PROMPT_TEXT, BOX_THRESHOLD, TEXT_THRESHOLD, models.device
            )

            if boxes.shape[0] > 0:
                boxes = clamp_boxes(boxes, crop_w, crop_h)
                keep = nms_xyxy(boxes, scores, NMS_IOU)
                boxes = boxes[keep]
                scores = scores[keep]
                phrases = [phrases[i] for i in keep]

            dets = []
            if boxes.shape[0] > 0:
                masks = sam_segment(models.sam_predictor, crop, boxes)
                sink_mask = polygon_to_mask(sink_poly_cropped, crop_h, crop_w)
                for box, sc, phr, m in zip(boxes, scores, phrases, masks):
                    ov = overlap_fraction(m, sink_mask)
                    dets.append((box, float(sc), str(phr), m, float(ov)))
            last = dets

        # build display
        display_crop = cv2.resize(crop, (VIEW_W, VIEW_H))

        items_in_sink = []
        items_outside = []

        for (box, sc, phr, m, ov) in last:
            bx1, by1, bx2, by2 = box
            # scale box to display coords
            dbx1, dby1 = int(bx1 * scale_x), int(by1 * scale_y)
            dbx2, dby2 = int(bx2 * scale_x), int(by2 * scale_y)

            inside = ov >= MASK_OVERLAP_THRESHOLD
            color = (0, 0, 255) if inside else (0, 255, 0)

            label = f"{phr} {sc:.2f} ov={ov:.2f}"
            (items_in_sink if inside else items_outside).append(label)

            # mask overlay (scale mask to display)
            m_disp = cv2.resize(m.astype(np.uint8), (VIEW_W, VIEW_H), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay = display_crop.copy()
            overlay[m_disp] = (overlay[m_disp] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            display_crop = overlay

            cv2.rectangle(display_crop, (dbx1, dby1), (dbx2, dby2), color, 2)
            cv2.putText(display_crop, label, (dbx1, max(15, dby1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # draw sink boundary
        cv2.polylines(display_crop, [sink_poly_scaled], True, (255, 255, 0), 2)

        # final UI (like yours)
        final_ui = np.zeros((VIEW_H, 800, 3), dtype=np.uint8)
        final_ui[:, 150:] = display_crop

        sidebar_w = 150
        cv2.rectangle(final_ui, (0, 0), (sidebar_w, VIEW_H), (20, 20, 20), -1)
        cv2.line(final_ui, (sidebar_w, 0), (sidebar_w, VIEW_H), (200, 200, 200), 1)

        cv2.putText(final_ui, f"FR: {frame_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(final_ui, "IN SINK", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
        y_pos = 85
        for item in items_in_sink[:10]:
            cv2.putText(final_ui, f"> {item[:28]}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_pos += 20

        cv2.putText(final_ui, "OUTSIDE", (10, 300), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
        y_pos = 325
        for item in items_outside[:10]:
            cv2.putText(final_ui, f"> {item[:28]}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 20

        cv2.imshow("Dish Monitor (DINO + SAM)", final_ui)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

# -------------- Main execution (YOLO-like) --------------
if __name__ == "__main__":
    if "--calibrate" in sys.argv or "-c" in sys.argv:
        from calibrate_sink import run_calibration
        run_calibration()
        sys.exit(0)

    if "--debug" in sys.argv or "-d" in sys.argv:
        DEBUG = True
        print("[DEBUG] enabled")

    # Optional: allow --device cpu|mps|cuda
    device = ""
    if "--device" in sys.argv:
        try:
            idx = sys.argv.index("--device")
            device = sys.argv[idx + 1]
        except Exception:
            print("Usage: python detection_dino_sam.py --device mps")
            sys.exit(1)

    device = choose_device(device)
    print(f"Loading GroundingDINO + SAM on device: {device}")
    models = load_models(device=device)
    print(f"Prompt: {PROMPT_TEXT}")

    video_list = []
    argv = sys.argv[1:]

    if "--videos" in argv or "-v" in argv:
        flag = "-v" if "-v" in argv else "--videos"
        i = argv.index(flag) + 1
        names = []
        while i < len(argv) and not argv[i].startswith("-"):
            names.append(argv[i])
            i += 1
        if names:
            video_list, missing = get_video_paths_by_names(names)
            if missing:
                print(f"Warning: not found in {DISH_VIDEOS_DIR}: {missing}")
            if not video_list:
                print("No valid video names. Available videos:")
                for f in list_videos_in_dish_videos():
                    print(f"  {f}")
                sys.exit(1)
        else:
            print("Usage: python detection_dino_sam.py --videos video1.mp4 video2.mp4")
            sys.exit(1)
    else:
        user_time = input("Enter start time (YYYY/MM/DD HH:MM) or video names (comma-separated) or press Enter for all: ").strip()
        if not user_time:
            video_list = get_videos_from_dish_videos(since_dt=None)
            print("Using all videos in dish_videos.")
        elif "," in user_time or user_time.endswith(".mp4") or not user_time[0].isdigit():
            names = [n.strip() for n in user_time.split(",") if n.strip()]
            video_list, missing = get_video_paths_by_names(names)
            if missing:
                print(f"Not found in {DISH_VIDEOS_DIR}: {missing}")
            if not video_list:
                print("Available:", ", ".join(list_videos_in_dish_videos()))
                sys.exit(1)
        else:
            try:
                since_dt = datetime.strptime(user_time, "%Y/%m/%d %H:%M")
                video_list = get_videos_from_dish_videos(since_dt=since_dt)
            except ValueError:
                print("Invalid format. Use YYYY/MM/DD HH:MM or comma-separated video names.")
                sys.exit(1)

    if not video_list:
        print(f"No videos found in {DISH_VIDEOS_DIR}.")
        sys.exit(1)

    print(f"Processing {len(video_list)} video(s). Press 'q' to quit window.")
    for path in video_list:
        print(f"--- {os.path.basename(path)} ---")
        process_video(path, models=models)

    cv2.destroyAllWindows()