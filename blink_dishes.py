import asyncio
import os
from aiohttp import ClientSession
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth
from blinkpy.helpers.util import json_load

# Configuration
CRED_FILE = "blink_auth.json"
OUTPUT_DIR = "./dish_videos"

async def setup_blink():
    # 1. Initialize Blink with a session
    blink = Blink(session=ClientSession())

    # 2. Check if we have saved credentials
    if os.path.exists(CRED_FILE):
        print(f"--- Loading saved credentials from {CRED_FILE} ---")
        # Load the saved credentials dictionary
        creds = await json_load(CRED_FILE)
        # Re-attach them to the auth handler
        blink.auth = Auth(creds, no_prompt=True)
    else:
        print("--- No saved credentials found. Starting first-time login ---")
        username = input("Enter Blink Email: ")
        password = input("Enter Blink Password: ")
        blink.auth = Auth({"username": username, "password": password})

    # 3. Start the system
    try:
        await blink.start()
    except Exception as e:
        # This catches the 2FA requirement
        if "2FA" in str(e) or "key" in str(e).lower():
            await blink.prompt_2fa()
        else:
            print(f"Login failed: {e}")
            return

    # 4. Save credentials for next time (Remote Cluster Use)
    if not os.path.exists(CRED_FILE):
        await blink.save(CRED_FILE)
        print(f"Credentials saved to {CRED_FILE}. Move this file to your cluster later!")

    # 5. Do something! (e.g., list cameras and download latest clip)
    print("\nConnected! Found Cameras:")
    for name, camera in blink.cameras.items():
        print(f"- {name} (Battery: {camera.battery})")
        
        # Example: Download the last motion clip
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        video_path = os.path.join(OUTPUT_DIR, f"{name}_latest.mp4")
        print(f"Downloading latest clip for {name} to {video_path}...")
        await camera.video_to_file(video_path)

    await blink.session.close()

if __name__ == "__main__":
    asyncio.run(setup_blink())