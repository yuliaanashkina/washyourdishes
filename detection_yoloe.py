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
PROMPT_CLASSES = ["cup", "bowl", "pan"]
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
    """
    Process one video and track dishes in/out of sink polygon.
    Uses YOLO-World model with prompt classes (cup, bowl, pan).
    """
    global sink_inventory
    clip_index = clip_index if clip_index is not None else 0
    clip_name = clip_name or os.path.basename(video_path)
    if model is None:
        model = YOLO(YOLOWORLD_MODEL)
        model.set_classes(PROMPT_CLASSES)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return {"in_sink_at_end": set(), "entered_this_clip": set(), "left_this_clip": set()}

    sink_polygon = get_sink_polygon()
    in_sink_at_start = set(sink_inventory.keys())
    entered_this_clip = set()
    left_this_clip = set()

    frame_idx = 0
    print(f"\n[Clip {clip_index}] Processing: {clip_name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame = cv2.resize(frame, (800, 600))
        current_time = time.time()
        # YOLO-World: no classes= arg; model already has set_classes(["cup","bowl","pan"])
        results = model.track(frame, persist=True, verbose=False)

        current_frame_ids = set()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, clss):
                tid = int(track_id)
                current_frame_ids.add(tid)
                x1, y1, x2, y2 = box
                xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
                center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                in_sink = is_point_in_sink(center_point, sink_polygon)

                class_name = _get_class_name(model, cls_id)

                if in_sink:
                    cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (0, 0, 255), 2)
                    if tid not in sink_inventory:
                        sink_inventory[tid] = {'entry_time': current_time, 'status': 'VISIBLE'}
                        entered_this_clip.add(tid)
                        if DEBUG:
                            print(f"  [DEBUG] frame {frame_idx} track_id={tid} {class_name} center={center_point} -> ENTERED sink")
                    elif DEBUG and frame_idx % 30 == 0:
                        print(f"  [DEBUG] frame {frame_idx} track_id={tid} {class_name} -> in sink (still)")
                else:
                    cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (0, 255, 0), 2)
                    if tid in sink_inventory:
                        left_this_clip.add(tid)
                        del sink_inventory[tid]
                        if DEBUG:
                            print(f"  [DEBUG] frame {frame_idx} track_id={tid} {class_name} center={center_point} -> LEFT sink")

                # Label: cup, bowl, pan + track id
                label = f"{class_name} #{tid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (xi1, yi1 - th - 4), (xi1 + tw, yi1), (0, 0, 0), -1)
                cv2.putText(frame, label, (xi1, yi1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw UI
        cv2.polylines(frame, [sink_polygon], True, (255, 255, 0), 2)
        cv2.imshow("YOLO-World Dish Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    in_sink_at_end = set(sink_inventory.keys())

    # Per-clip summary
    print(f"\n--- After clip {clip_index} ({clip_name}) ---")
    if in_sink_at_end:
        for tid in sorted(in_sink_at_end):
            print(f"  Dish {tid} is in the set.")
    else:
        print("  No dishes in the set.")
    if entered_this_clip:
        for tid in sorted(entered_this_clip):
            print(f"  Dish {tid} entered the set during this clip.")
    if left_this_clip:
        for tid in sorted(left_this_clip):
            print(f"  Dish {tid} left the set during this clip.")
    print(f"  Summary: in_sink_now={sorted(in_sink_at_end)} | entered_this_clip={sorted(entered_this_clip)} | left_this_clip={sorted(left_this_clip)}")
    if DEBUG:
        print(f"  [DEBUG] in_sink_at_start={sorted(in_sink_at_start)}")

    return {
        "in_sink_at_end": in_sink_at_end,
        "entered_this_clip": entered_this_clip,
        "left_this_clip": left_this_clip,
    }

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
