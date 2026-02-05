# TO DO LIST 

Figure out how to get the video frames "blink" app into the script (likely upload the videos after they are saved or Blinkpy)

implemment the idea that there is no time limit to flag the clip, but we flag a clip anytime an object A in the sink set stays there until the end of the day and is not removed. so we can find the clip where object A was put in.

use the blinkpy library to have the clips send after they are saved.

Draw the polygon on the sink, based on where camera is.

Test the cup and bowl moving out of the polygon.


# SUMMARY of the detections script
1. The Core Logic: "Strict Boundary" Rules

  Entry (Check-In): If a dish (cup or bowl) is detected inside the sink polygon, it is added to the sink_inventory and a timer starts.

  The "Hidden" State (Object Permanence): If a dish is in the inventory but the camera stops seeing it (e.g., it gets covered by another plate), the system does not remove it. It marks the status as COVERED and keeps the timer running.

  Exit (Check-Out): A dish is removed from the inventory ONLY if the camera actively sees that specific ID outside the sink polygon (e.g., placed on the counter). Simply disappearing is not enough to be removed.

2. The Tech Pipeline
  Input: Accepts raw video feeds (RTSP, Webcam, or File). No encoding is required, but resizing high-res streams to ~720p is recommended for performance.

  Detection: Uses YOLOv8 (Nano) to identify objects frame-by-frame.

  Tracking: Uses BoT-SORT to assign unique IDs (e.g., "Plate #42"), ensuring the system tracks individual items across time.

3. Key Constraints & Features
  Duration: The script can run indefinitely (24/7). It uses a "Garbage Collector" to delete "ghost" items that have been hidden for >1 hour to prevent memory bloat.

  Alerting: If Current Time - Entry Time > 10 minutes, the status changes to ALERT (Visual Red Box/Text).

  Limitations: The "Strict Exit" rule means if a user blocks the camera's view while removing a plate, the system will falsely believe the plate is still in the sink (hidden).

# Current Edgecases or things to Consider:


How to have the camera live feed be cast real time to our detection

When a plate/bowl/cup stay in sink for >=10min, how to pull the time stamp (either current time stamp or when the object was placed in the sink)

Need to adjust the polygon to match the sink based on the camera

Do we need to show the objects in the sink moving to the counter, like does the counter need to be in frame??
