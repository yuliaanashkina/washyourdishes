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
# COCO class IDs: 41=cup, 45=bowl.
TARGET_CLASSES = [41, 45]  
ALERT_THRESHOLD = 5
DISH_VIDEOS_DIR = "./dish_videos"
SINK_POLYGON_FILE = "sink_polygon.json"
DEBUG = False 

BLINK_FNAME_PATTERN = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})t(\d{2})-(\d{2})-(\d{2})", re.IGNORECASE
)

# --- HELPERS ---

def _get_class_name(model, cls_id):
    """Helper to get class name from model metadata. Fixed NameError."""
    try:
        return model.names[int(cls_id)]
    except (AttributeError, KeyError):
        return f"ID {cls_id}"

def list_videos_in_dish_videos():
    if not os.path.isdir(DISH_VIDEOS_DIR):
        return []
    return sorted(
        f for f in os.listdir(DISH_VIDEOS_DIR)
        if f.lower().endswith(".mp4")
    )

def get_video_paths_by_names(names):
    if not os.path.isdir(DISH_VIDEOS_DIR):
        return [], list(names)
    available = set(list_videos_in_dish_videos())
    resolved = []
    missing = []
    for n in names:
        n = n.strip()
        if not n: continue
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

# --- SINK LOGIC ---
sink_inventory = {}  # track_id -> {entry_time, status}

def get_sink_polygon():
    if os.path.isfile(SINK_POLYGON_FILE):
        with open(SINK_POLYGON_FILE) as f:
            points = json.load(f)
        return np.array(points, np.int32)
    return np.array([[150, 100], [500, 100], [500, 380], [150, 380]], np.int32)

def is_point_in_sink(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def process_video(video_path, clip_index=None, clip_name=None, model=None):
    global sink_inventory
    clip_index = clip_index if clip_index is not None else 0
    clip_name = clip_name or os.path.basename(video_path)

    # Note: If you want to use YOLO-World, use model = YOLO('yolov8s-world.pt')
    # and call model.set_classes(...) here.

    cap = cv2.VideoCapture(video_path)
    sink_polygon = get_sink_polygon()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # Skip frames to speed up processing
        if frame_idx % 4 != 0: continue 

        frame = cv2.resize(frame, (800, 600))
        
        # 1. Tracker Call
        results = model.track(frame, persist=True, verbose=False, conf=0.1, iou=0.5)

        items_in_sink = []
        items_outside = []

        # 2. Logic: Process detections if they exist
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()

            for box, tid, cls_id in zip(boxes, track_ids, clss):
                # Only process classes defined in TARGET_CLASSES (Cup/Bowl)
                if int(cls_id) not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = box
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                in_sink = is_point_in_sink(center, sink_polygon)
                name = _get_class_name(model, cls_id)
                
                label = f"{name} #{tid}"
                if in_sink:
                    items_in_sink.append(label)
                    sink_inventory[tid] = {'status': 'active'}
                    color = (0, 0, 255) # Red for "Dirty/In Sink"
                else:
                    items_outside.append(label)
                    if tid in sink_inventory: del sink_inventory[tid]
                    color = (0, 255, 0) # Green for "Clean/Away"

                # Draw boxes on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 3. DRAW UI SIDEBAR
        sidebar_w = 220
        cv2.rectangle(frame, (0, 0), (sidebar_w, 600), (0, 0, 0), -1)
        cv2.line(frame, (sidebar_w, 0), (sidebar_w, 600), (255, 255, 255), 1)

        cv2.putText(frame, f"FRAME: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, "--- IN SINK ---", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for i, item in enumerate(items_in_sink[:8]):
            cv2.putText(frame, item, (15, 100 + (i*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "--- OUTSIDE ---", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for i, item in enumerate(items_outside[:8]):
            cv2.putText(frame, item, (15, 350 + (i*25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw sink boundary
        cv2.polylines(frame, [sink_polygon], True, (255, 255, 0), 2)

        cv2.imshow("Dish Monitor Debugger", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if "--calibrate" in sys.argv or "-c" in sys.argv:
        from calibrate_sink import run_calibration
        run_calibration()
        sys.exit(0)

    if "--debug" in sys.argv or "-d" in sys.argv:
        DEBUG = True
        print("[DEBUG] Verbose per-frame logging enabled.")

    video_list = []
    argv = sys.argv[1:]
    
    if "--videos" in argv or "-v" in argv:
        flag = "-v" if "-v" in argv else "--videos"
        idx = argv.index(flag) + 1
        names = []
        while idx < len(argv) and not argv[idx].startswith("-"):
            names.append(argv[idx])
            idx += 1
        if names:
            video_list, missing = get_video_paths_by_names(names)
            if missing:
                print(f"Warning: not found: {missing}")
        else:
            print("Usage: python detection.py --videos video1.mp4")
            sys.exit(1)
    else:
        msg = "Enter start time (YYYY/MM/DD HH:MM) or video names (comma-separated) or Enter for all: "
        user_time = input(msg).strip()
        if not user_time:
            video_list = get_videos_from_dish_videos(since_dt=None)
            print("Using all videos.")
        elif "," in user_time or user_time.endswith(".mp4") or not user_time[0].isdigit():
            names = [n.strip() for n in user_time.split(",") if n.strip()]
            video_list, missing = get_video_paths_by_names(names)
        else:
            try:
                since_dt = datetime.strptime(user_time, "%Y/%m/%d %H:%M")
                video_list = get_videos_from_dish_videos(since_dt=since_dt)
            except ValueError:
                print("Invalid format.")
                sys.exit(1)

    if not video_list:
        print(f"No videos found in {DISH_VIDEOS_DIR}.")
        sys.exit(1)

    print(f"Processing {len(video_list)} video(s).")
    # Using standard nano model for speed
    model = YOLO('yolov8n.pt') 
    
    for i, path in enumerate(video_list, start=1):
        name = os.path.basename(path)
        process_video(path, clip_index=i, clip_name=name, model=model)

    print("\n=== Final summary ===")
    if sink_inventory:
        print(f"Dishes currently in sink (ID numbers): {sorted(sink_inventory.keys())}")
    else:
        print("No dishes in the sink.")
    cv2.destroyAllWindows()