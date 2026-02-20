"""
Draw the sink polygon on a frame from a dish_videos MP4.
Left-click: add point. C: close & save. U: undo. R: reset. Q: quit without saving.
Saves to sink_polygon.json (800x600 coords) for use by detection.py.
"""
import cv2
import json
import os

DISH_VIDEOS_DIR = "./dish_videos"
SINK_POLYGON_FILE = "sink_polygon.json"
FRAME_SIZE = (800, 600)

WINDOW_NAME = "Calibrate Sink - click points, C to close & save, Q to quit"


def get_video_list():
    if not os.path.isdir(DISH_VIDEOS_DIR):
        return []
    files = [
        os.path.join(DISH_VIDEOS_DIR, f)
        for f in sorted(os.listdir(DISH_VIDEOS_DIR))
        if f.lower().endswith(".mp4")
    ]
    return files


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.resize(frame, FRAME_SIZE)


# Global state for mouse callback
points = []
close_requested = False


def mouse_callback(event, x, y, flags, param):
    global points, close_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
    if event == cv2.EVENT_RBUTTONDOWN:
        close_requested = True


def redraw(frame_display):
    display = frame_display.copy()
    n = len(points)
    for i, pt in enumerate(points):
        cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
        if i > 0:
            cv2.line(display, tuple(points[i - 1]), tuple(pt), (0, 255, 0), 2)
    if n >= 2:
        cv2.line(display, tuple(points[-1]), tuple(points[0]), (0, 255, 255), 1)
    cv2.putText(
        display,
        "Left-click: add point | C: close & save | U: undo | R: reset | Q: quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.imshow(WINDOW_NAME, display)


def run_calibration():
    global points, close_requested
    points = []
    close_requested = False

    videos = get_video_list()
    if not videos:
        print(f"No .mp4 files in {DISH_VIDEOS_DIR}. Run blink_dishes.py first.")
        return

    print(f"Found {len(videos)} video(s).")
    choice = input("Enter path or press Enter for first video: ").strip()
    video_path = choice if choice and os.path.isfile(choice) else videos[0]
    if not os.path.isfile(video_path):
        print("Invalid path. Using first video.")
        video_path = videos[0]

    frame = get_first_frame(video_path)
    if frame is None:
        print("Could not read first frame.")
        return

    print("Left-click: add point. C: close & save. U: undo. R: reset. Q: quit without saving.")
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    redraw(frame)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if close_requested or key == ord("c"):
            if len(points) < 3:
                if key == ord("c"):
                    print("Add at least 3 points before closing.")
                continue
            with open(SINK_POLYGON_FILE, "w") as f:
                json.dump(points, f, indent=2)
            cv2.destroyAllWindows()
            print(f"Saved {len(points)} points to {SINK_POLYGON_FILE}")
            return
        if key == ord("q"):
            cv2.destroyAllWindows()
            print("Quit without saving.")
            return
        if key == ord("u"):
            if points:
                points.pop()
        if key == ord("r"):
            points.clear()
        redraw(frame)


if __name__ == "__main__":
    run_calibration()
