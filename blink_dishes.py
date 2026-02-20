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
DEFAULT_SINCE_HOURS = 1  # 7 days; increase if you need older clips

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

    # 5. Refresh and sync camera data to get latest clips
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

    # Calculate time window once
    since_dt = datetime.now() - timedelta(hours=DEFAULT_SINCE_HOURS)
    
    # Check available clips for debugging (with error handling)
    print(f"\nChecking for clips since {since_dt.strftime('%Y/%m/%d %H:%M')}...")
    total_clips_found = 0
    for name, camera in blink.cameras.items():
        try:
            if hasattr(camera, 'clips') and camera.clips:
                # Try to filter by created_at if available
                try:
                    clips = [c for c in camera.clips if hasattr(c, 'created_at') and c.created_at >= since_dt]
                except (TypeError, AttributeError):
                    # If comparison fails, just count all clips
                    clips = list(camera.clips) if camera.clips else []
                if clips:
                    print(f"  {name}: {len(clips)} clip(s) found")
                    total_clips_found += len(clips)
                else:
                    print(f"  {name}: 0 clips found in time range")
            else:
                print(f"  {name}: No clips metadata available")
        except Exception as e:
            print(f"  {name}: Error checking clips: {e}")
    if total_clips_found == 0:
        print(f"Warning: No clips found in the last {DEFAULT_SINCE_HOURS} hour(s).")
        print("  Try increasing DEFAULT_SINCE_HOURS or check the Blink app for available clips.")
    since_str = since_dt.strftime("%Y/%m/%d %H:%M")
    now_str = datetime.now().strftime("%Y/%m/%d %H:%M")
    
    # Fix: show hours correctly, not days
    hours_text = f"{DEFAULT_SINCE_HOURS} hour(s)" if DEFAULT_SINCE_HOURS < 24 else f"{DEFAULT_SINCE_HOURS / 24:.1f} day(s)"
    
    print(f"\nQuerying clips from {since_str} to {now_str} (last {hours_text})")
    print(f"Clips will be saved to: {OUTPUT_DIR}")
    
    # Count existing files before download
    existing_count = sum(1 for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4"))
    if existing_count > 0:
        print(f"Note: {existing_count} existing .mp4 file(s) in directory (download_videos may skip existing files)")
    print()

    try:
        await blink.download_videos(OUTPUT_DIR, since=since_str, camera="all", delay=1)
        final_count = sum(1 for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4"))
        new_count = final_count - existing_count
        print(f"\nDone. {final_count} total clip(s) in {OUTPUT_DIR} ({new_count} new)")
        if new_count == 0 and existing_count > 0:
            print("Warning: No new clips downloaded. This could mean:")
            print("  - All clips in this time range were already downloaded")
            print("  - No new clips exist in this time range")
            print("  - Try increasing DEFAULT_SINCE_HOURS or checking the Blink app for available clips")
    except Exception as e:
        print(f"Download error: {e}")
        print("If you see no clips: check your Blink subscription (cloud clip retention) and that clips exist in the app for this period.")

    # 7. Close the session
    await session.close()

if __name__ == "__main__":
    asyncio.run(setup_blink())