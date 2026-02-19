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

        # 5. Refresh data 
        print("Refreshing camera data...")
        await blink.refresh()
        
        # --- NEW: Aggressive Video Sync ---
        print("Synchronizing video manifest...")
        # This reaches into the internal sync module logic to find clips
        for sync_name, sync_module in blink.sync.items():
            await sync_module.refresh() 
        # ----------------------------------

        print("\nConnected! Found Cameras:")
        for name, camera in blink.cameras.items():
            print(f"- {name}")
            
            # Check the library's internal 'last_record' attribute
            # Sometimes 'camera.clip' is empty but the last_record is populated
            last_clip = getattr(camera, 'last_record', None)
            print(f"  > Camera Clip Attribute: {camera.clip}")
            print(f"  > Last Record Attribute: {last_clip}")

            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            
            print(f"  > Attempting download...")
            # We use 'stop=5' to look back at the last 5 clips 
            # in case the 'latest' one is still processing.
            await blink.download_videos(
                OUTPUT_DIR, 
                stop=5, 
                camera=name, 
                delay=2
            )
            
            # Check if files appeared
            if os.listdir(OUTPUT_DIR):
                print(f"  > Success! Files found: {os.listdir(OUTPUT_DIR)}")
            else:
                print(f"  > Still no files. This suggests the API is not sharing the manifest.")

    except Exception as e:
        print(f"\n[!] AN ERROR OCCURRED: {e}")
    
    finally:
        # 6. This runs no matter what, preventing the "Unclosed session" warning
        print("\n--- Closing session safely ---")
        await session.close()

if __name__ == "__main__":
    asyncio.run(setup_blink())