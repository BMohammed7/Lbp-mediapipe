import cv2
import mediapipe as mp
import urllib.request
import os

# 1. Ensure the modern AI model is downloaded
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe AI Model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)

# 2. Setup the modern Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Standard Skeleton Connections
POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10), 
    (11,12), (11,13), (13,15), (15,17), (15,19), (15,21), (17,19), 
    (12,14), (14,16), (16,18), (16,20), (16,22), (18,20), (11,23), 
    (12,24), (23,24), (23,25), (25,27), (27,29), (27,31), (29,31), 
    (24,26), (26,28), (28,30), (28,32), (30,32)
]

# --- IMPORTANT: CHANGE THIS TO YOUR ACTUAL VIDEO FILE NAME ---
input_video_path = 'JumpingJacks.mp4' 
output_video_path = 'tracked_output.mp4'

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open '{input_video_path}'. Make sure the name is exactly right and it is in the same folder.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps <= 0: fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Processing '{input_video_path}'... A window will pop up showing the progress.")

# 3. Process the video frame-by-frame
with PoseLandmarker.create_from_options(options) as landmarker:
    frame_timestamp_ms = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break # Reached the end of the video

        # MediaPipe requires a strict timeline for video mode
        frame_timestamp_ms += int(1000 / fps)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect the pose on this specific frame
        pose_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw the stick figure manually
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

        # Save the drawn frame to the new video
        out.write(frame)
        cv2.imshow('Processing Video...', frame)

        # Press 'q' if you want to cancel early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing cancelled.")
            break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished! The tracked video is saved in your folder as '{output_video_path}'.")