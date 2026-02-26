"""
Detection script using YOLO-World (open-vocabulary) with text prompts:
cups, bowls, and pans. Same sink-tracking logic as detection.py.
"""
import cv2
import json
import numpy as np
import time
import os
import re
import sys
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURATION ---
# YOLO-World: prompt-based classes (no COCO IDs). Detects cups, bowls, pans.
PROMPT_CLASSES = [
    "blue plastic bowl",
    "white ceramic coffee mug",
    "blue ceramic mug",
    "clear glass cup",
    "stainless steel pot",
    "black frying pan",
    "silver metal pot lid",
    "clear plastic food container",
    "silver metal spoon",
    "silver metal fork",
    "rectangular glass baking dish",
    "white ceramic plate"
]
ALERT_THRESHOLD = 5
DISH_VIDEOS_DIR = "./dish_videos"
SINK_POLYGON_FILE = "sink_polygon.json"
DEBUG = False  # Set True or use --debug for verbose per-frame logging
# YOLO-World model (s=small, m=medium, l=large). First run downloads weights.
YOLOWORLD_MODEL = "yolov8s-world.pt"

# Blink filenames look like: mini-2-0eew-2026-02-19t01-21-26-00-00.mp4
BLINK_FNAME_PATTERN = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})t(\d{2})-(\d{2})-(\d{2})", re.IGNORECASE
)


def list_videos_in_dish_videos():
    """Return sorted list of .mp4 filenames (basenames only) in DISH_VIDEOS_DIR."""
    if not os.path.isdir(DISH_VIDEOS_DIR):
        return []
    return sorted(
        f for f in os.listdir(DISH_VIDEOS_DIR)
        if f.lower().endswith(".mp4")
    )


def get_video_paths_by_names(names):
    """
    Resolve video names to full paths under DISH_VIDEOS_DIR.
    Names can be full filenames (e.g. tuzar_dish_in1.mp4) or paths; missing files are reported.
    Returns (resolved_paths, missing_names).
    """
    if not os.path.isdir(DISH_VIDEOS_DIR):
        return [], list(names)
    available = set(list_videos_in_dish_videos())
    resolved = []
    missing = []
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
    """
    Returns sorted list of .mp4 paths in DISH_VIDEOS_DIR.
    If since_dt is given, only includes files with timestamp >= since_dt (from filename or mtime).
    """
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

# --- SINK LOGIC ---
sink_inventory = {}  # track_id -> {entry_time, status}; persisted across clips

def get_sink_polygon():
    if os.path.isfile(SINK_POLYGON_FILE):
        with open(SINK_POLYGON_FILE) as f:
            points = json.load(f)
        return np.array(points, np.int32)
    return np.array([[150, 100], [500, 100], [500, 380], [150, 380]], np.int32)

def is_point_in_sink(point, polygon):
    return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0


def _get_class_name(model, cls_id):
    """Resolve class index to name (works for both dict and list model.names)."""
    cid = int(cls_id)
    names = model.names
    if isinstance(names, dict):
        return names.get(cid, f"class_{cid}")
    if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
        return names[cid]
    return f"class_{cid}"

def process_video(video_path, clip_index=None, clip_name=None, model=None):
    global sink_inventory
    clip_index = clip_index if clip_index is not None else 0
    clip_name = clip_name or os.path.basename(video_path)

    if model is None:
        model = YOLO(YOLOWORLD_MODEL)
        model.set_classes(PROMPT_CLASSES)

    # --- 1. PREPARE LIGHTING NORMALIZATION (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    cap = cv2.VideoCapture(video_path)
    original_sink_poly = get_sink_polygon()
    
    # --- 2. CALCULATE CROP BOX ---
    sx, sy, sw, sh = cv2.boundingRect(original_sink_poly)
    pad = 60 
    x1, y1 = max(0, sx - pad), max(0, sy - pad)
    x2, y2 = min(800, sx + sw + pad), min(600, sy + sh + pad)
    
    sink_poly_cropped = original_sink_poly - [x1, y1]

    view_w, view_h = 650, 600
    scale_x, scale_y = view_w / (x2 - x1), view_h / (y2 - y1)
    sink_poly_scaled = (sink_poly_cropped * [scale_x, scale_y]).astype(np.int32)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % 4 != 0: continue 

        frame = cv2.resize(frame, (800, 600))
        crop_img = frame[y1:y2, x1:x2].copy()
        
        lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        normalized_crop = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        display_crop = cv2.resize(normalized_crop, (view_w, view_h))
        
        # --- 3. RUN TRACKER ---
        results = model.track(normalized_crop, imgsz=1280, persist=True, verbose=False, conf=0.1, iou=0.5)

        # 4. UPDATED STICKY LOGIC: Store both Name and Zone
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            tids = results[0].boxes.id.int().cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()

            for box, tid, cls_id in zip(boxes, tids, clss):
                bx1, by1, bx2, by2 = box
                center = (int((bx1 + bx2) / 2), int((by1 + by2) / 2))
                in_sink = is_point_in_sink(center, sink_poly_cropped)
                
                # Get the descriptive name (e.g., "blue plastic bowl")
                name = _get_class_name(model, cls_id)
                
                # Update inventory with the description and current zone
                sink_inventory[tid] = {"name": name, "zone": "in_sink" if in_sink else "outside"}

                # Draw rectangles for currently visible items
                dbx1, dby1 = int(bx1 * scale_x), int(by1 * scale_y)
                dbx2, dby2 = int(bx2 * scale_x), int(by2 * scale_y)
                color = (0, 0, 255) if in_sink else (0, 255, 0)
                cv2.rectangle(display_crop, (dbx1, dby1), (dbx2, dby2), color, 2)
                cv2.putText(display_crop, f"{name} #{tid}", (dbx1, dby1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 5. UI RENDERING: Use the stored names for the sidebar
        items_in_sink = [f"{data['name']} #{tid}" for tid, data in sink_inventory.items() if data['zone'] == "in_sink"]
        items_outside = [f"{data['name']} #{tid}" for tid, data in sink_inventory.items() if data['zone'] == "outside"]

        # Final UI Assembly
        final_ui = np.zeros((600, 800, 3), dtype=np.uint8)
        final_ui[:, 150:] = display_crop
        
        sidebar_w = 150
        cv2.rectangle(final_ui, (0, 0), (sidebar_w, 600), (20, 20, 20), -1)
        cv2.line(final_ui, (sidebar_w, 0), (sidebar_w, 600), (200, 200, 200), 1)

        cv2.putText(final_ui, f"FR: {frame_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(final_ui, "IN SINK", (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)
        y_pos = 85
        for item in items_in_sink[:12]:
            cv2.putText(final_ui, f"> {item}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            y_pos += 20

        cv2.putText(final_ui, "OUTSIDE", (10, 350), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
        y_pos = 375
        for item in items_outside[:10]:
            cv2.putText(final_ui, f"> {item}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_pos += 20

        cv2.polylines(display_crop, [sink_poly_scaled], True, (255, 255, 0), 2)
        cv2.imshow("Dish Monitor Debugger", final_ui)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if "--calibrate" in sys.argv or "-c" in sys.argv:
        from calibrate_sink import run_calibration
        run_calibration()
        sys.exit(0)

    if "--debug" in sys.argv or "-d" in sys.argv:
        globals()["DEBUG"] = True
        print("[DEBUG] Verbose per-frame logging enabled.")

    # Build model once with prompt classes (cups, bowls, pans)
    print(f"Loading YOLO-World ({YOLOWORLD_MODEL}) with prompts: {PROMPT_CLASSES}")
    model = YOLO(YOLOWORLD_MODEL)
    model.set_classes(PROMPT_CLASSES)

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
            print("Usage: python detection_yoloe.py --videos video1.mp4 video2.mp4 video3.mp4")
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

    print(f"Processing {len(video_list)} video(s) in order (track IDs persist across clips).")
    for i, path in enumerate(video_list, start=1):
        name = os.path.basename(path)
        process_video(path, clip_index=i, clip_name=name, model=model)

    print("\n=== Final summary ===")
    if sink_inventory:
        print(f"Dishes currently in the set: {sorted(sink_inventory.keys())}")
    else:
        print("No dishes in the set.")
    cv2.destroyAllWindows()
