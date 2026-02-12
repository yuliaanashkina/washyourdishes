import cv2
import numpy as np
import time
import asyncio
import os
from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth
from blinkpy.helpers.util import json_load
from ultralytics import YOLO

# --- CONFIGURATION ---
TARGET_CLASSES = [41, 45]  # COCO IDs: Cup, Bowl
ALERT_THRESHOLD = 5        
BLINK_CRED_FILE = "blink_auth.json"
DOWNLOAD_DIR = "./blink_clips"
CAMERA_NAME = "Kitchen"  # Change this to your exact Blink camera name

# --- BLINK DOWNLOAD METHOD ---
async def download_recent_clips(since_time_str):
    """
    Downloads clips from Blink since a specific time.
    Format for since_time_str: "2026/02/11 15:30"
    """
    async with ClientSession() as session:
        blink = Blink(session=session)
        
        # Load saved credentials
        if not os.path.exists(BLINK_CRED_FILE):
            print(f"Error: {BLINK_CRED_FILE} not found. Run local login script first.")
            return None
        
        creds = await json_load(BLINK_CRED_FILE)
        blink.auth = Auth(creds, no_prompt=True)
        
        await blink.start()
        
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)

        print(f"Searching for clips since {since_time_str}...")
        # This downloads all clips since the timestamp for the specific camera
        await blink.download_videos(DOWNLOAD_DIR, since=since_time_str, camera=CAMERA_NAME)
        
        # Get list of downloaded files
        files = [os.path.join(DOWNLOAD_DIR, f) for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.mp4')]
        return sorted(files) # Process oldest to newest

# --- SINK LOGIC ---
sink_inventory = {}

def get_sink_polygon():
    return np.array([[150, 100], [500, 100], [500, 380], [150, 380]], np.int32)

def is_point_in_sink(point, polygon):
    return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0

def process_video(video_path):
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(video_path)
    sink_polygon = get_sink_polygon()
    
    print(f"Processing: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (800, 600))
        current_time = time.time()
        results = model.track(frame, persist=True, classes=TARGET_CLASSES, verbose=False)
        
        current_frame_ids = set()
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                current_frame_ids.add(track_id)
                x1, y1, x2, y2 = box
                center_point = (int((x1+x2)/2), int((y1+y2)/2))
                
                if is_point_in_sink(center_point, sink_polygon):
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    if track_id not in sink_inventory:
                        sink_inventory[track_id] = {'entry_time': current_time, 'status': 'VISIBLE'}
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    if track_id in sink_inventory:
                        del sink_inventory[track_id]

        # Draw UI
        cv2.polylines(frame, [sink_polygon], True, (255, 255, 0), 2)
        cv2.imshow("Blink Dish Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Ask for time window
    user_time = input("Enter start time (YYYY/MM/DD HH:MM) or press Enter for last 1hr: ")
    
    if not user_time:
        # Default to 1 hour ago
        from datetime import datetime, timedelta
        user_time = (datetime.now() - timedelta(hours=1)).strftime("%Y/%m/%d %H:%M")

    # 2. Download from Blink
    video_list = asyncio.run(download_recent_clips(user_time))

    # 3. Process each video found
    if video_list:
        print(f"Found {len(video_list)} videos. Starting AI analysis...")
        for video in video_list:
            process_video(video)
    else:
        print("No videos found for that time period.")