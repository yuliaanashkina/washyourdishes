import asyncio
import os
from datetime import datetime, timedelta
from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth
from blinkpy.helpers.util import json_load

# Configuration
CRED_FILE = "blink_auth.json"
OUTPUT_DIR = "./dish_videos"
# How far back to look for clips (Blink app shows these; API returns them by time)
DEFAULT_SINCE_MINUTES = 10

async def setup_blink():
    # 1. Initialize session
    session = ClientSession()
    # Using a try block so the finally block ALWAYS closes the session
    try:
        blink = Blink(session=session)

        # 2. Check credentials
        if os.path.exists(CRED_FILE):
            print(f"--- Loading saved credentials from {CRED_FILE} ---")
            creds = await json_load(CRED_FILE)
            blink.auth = Auth(creds, no_prompt=True)
        else:
            print("--- No saved credentials found ---")
            username = input("Enter Blink Email: ")
            password = input("Enter Blink Password: ")
            blink.auth = Auth({"username": username, "password": password})

        # 3. Start system
        await blink.start()

        # 4. Save credentials if new
        if not os.path.exists(CRED_FILE):
            await blink.save(CRED_FILE)
            print(f"Credentials saved to {CRED_FILE}.")

    # 5. Refresh and list cameras
    print("Refreshing camera data...")
    await blink.refresh()

    print("\nConnected! Found Cameras:")
    for name, camera in blink.cameras.items():
        print(f"  - {name} (Battery: {camera.battery})")

    # 6. Download clips using the time-based API (same clips you see in the Blink app)
    # video_to_file() often fails with "no saved video" because Blink's API doesn't
    # populate the per-camera "last clip" field; download_videos(since=...) works.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    since_dt = datetime.now() - timedelta(minutes=DEFAULT_SINCE_MINUTES)
    since_str = since_dt.strftime("%Y/%m/%d %H:%M")
    print(f"\nDownloading clips since {since_str} (last {DEFAULT_SINCE_MINUTES} minutes)...")
    print(f"Clips will be saved to: {OUTPUT_DIR}\n")

    try:
        await blink.download_videos(OUTPUT_DIR, since=since_str, camera="all", delay=1)
        count = sum(1 for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4"))
        print(f"\nDone. {count} clip(s) in {OUTPUT_DIR}")
    except Exception as e:
        print(f"Download error: {e}")
        print("If you see no clips: check your Blink subscription (cloud clip retention) and that clips exist in the app for this period.")

    # 7. Close the session
    await session.close()

if __name__ == "__main__":
    asyncio.run(setup_blink())