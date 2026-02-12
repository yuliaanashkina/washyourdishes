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
    # 1. Initialize session separately so we can close it correctly later
    session = ClientSession()
    blink = Blink(session=session)

    # 2. Check if we have saved credentials
    if os.path.exists(CRED_FILE):
        print(f"--- Loading saved credentials from {CRED_FILE} ---")
        creds = await json_load(CRED_FILE)
        blink.auth = Auth(creds, no_prompt=True)
    else:
        print("--- No saved credentials found. Starting first-time login ---")
        username = input("Enter Blink Email: ")
        password = input("Enter Blink Password: ")
        blink.auth = Auth({"username": username, "password": password})

    # 3. Start the system with improved 2FA handling
    try:
        await blink.start()
    except Exception as e:
        print(f"Initial connection result: {e}")
        # Force the 2FA prompt if we aren't fully started
        if not blink.available:
            print("Attempting 2FA verification...")
            await blink.prompt_2fa()
            # Re-start after 2FA
            await blink.start()
        else:
            print(f"Login failed: {e}")
            await session.close()
            return

    # 4. Save credentials for next time
    if not os.path.exists(CRED_FILE):
        await blink.save(CRED_FILE)
        print(f"Credentials saved to {CRED_FILE}. Move this file to your cluster later!")

    # 5. Refresh data to ensure video URLs are populated
    print("Refreshing camera data...")
    await blink.refresh()

    print("\nConnected! Found Cameras:")
    for name, camera in blink.cameras.items():
        print(f"- {name} (Battery: {camera.battery})")
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        video_path = os.path.join(OUTPUT_DIR, f"{name}_latest.mp4")
        print(f"Downloading latest clip for {name} to {video_path}...")
        
        # Download the last motion clip if available
        try:
            await camera.video_to_file(video_path)
        except Exception as video_err:
            print(f"Could not download video for {name}: {video_err}")

    # 6. Close the session using the session variable
    await session.close()

if __name__ == "__main__":
    asyncio.run(setup_blink())