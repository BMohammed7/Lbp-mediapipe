import cv2
import mediapipe as mp
import time
import urllib.request
import os

# 1. Download the required AI model automatically (Only happens once)
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Downloading modern MediaPipe AI Model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Setup the modern Tasks API (bypasses the broken 'solutions' API entirely)
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Standard Skeleton Connections to draw the Stick Figure
POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), 
    (11,12), (11,13), (13,15), (15,17), (15,19), (15,21), (17,19), 
    (12,14), (14,16), (16,18), (16,20), (16,22), (18,20), (11,23), 
    (12,24), (23,24), (23,25), (25,27), (27,29), (27,31), (29,31), 
    (24,26), (26,28), (28,30), (28,32), (30,32)
]

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0: fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('lifting_capture.mp4', fourcc, fps, (frame_width, frame_height))

print("Starting 30-second recording... Perform your lifting motion now!")
start_time = time.time()
record_duration = 30

# 3. Run the tracking
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        timestamp_ms = int((time.time() - start_time) * 1000)
        if timestamp_ms <= 0: timestamp_ms = 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process the frame to find joints
        pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Manually draw the stick figure (Bypasses broken drawing tools)
        if pose_result.pose_landmarks:
            for pose_landmarks in pose_result.pose_landmarks:
                # Draw the bones (white lines)
                for start_idx, end_idx in POSE_CONNECTIONS:
                    p1 = pose_landmarks[start_idx]
                    p2 = pose_landmarks[end_idx]
                    if p1.visibility > 0.5 and p2.visibility > 0.5:
                        x1, y1 = int(p1.x * frame_width), int(p1.y * frame_height)
                        x2, y2 = int(p2.x * frame_width), int(p2.y * frame_height)
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
                
                # Draw the joints (red dots)
                for lm in pose_landmarks:
                    if lm.visibility > 0.5:
                        x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        out.write(frame)
        
        elapsed_time = time.time() - start_time
        time_left = max(0, record_duration - elapsed_time)
        
        cv2.putText(frame, f"Recording: {time_left:.1f}s left", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Modern MediaPipe Tracking', frame)

        if elapsed_time > record_duration or cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording complete. Saved as 'lifting_capture.mp4'.")
            break

cap.release()
out.release()
cv2.destroyAllWindows()