import cv2
import numpy as np
import onnxruntime as ort

# 1. Load the AI Model (runs instantly in Python)
print("Loading model...")
session = ort.InferenceSession("spinepose.onnx")
input_name = session.get_inputs()[0].name
print("Model loaded!")

# 2. Open your video
video_path = "JumpingJacks.mp4" # Change to your video name
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for the AI (192x256)
    img_resized = cv2.resize(frame, (192, 256))
    
    # Convert image to math tensor (NCHW format)
    img_data = img_resized.astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1)) # Move colors to front
    img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension

    # 3. Run Inference
    results = session.run(None, {input_name: img_data})
    
    # Grab the X and Y coordinates (SimCC format)
    simcc_x = results[0][0] # First output
    simcc_y = results[1][0] # Second output


   # 4. Draw ONLY the 9 spine points on the frame
    h, w, _ = frame.shape
    spine_points = []

    # The exact indices for the 9 vertebrae points based on your debug map
    spine_indices = [28, 29, 30, 31, 32, 33, 34, 35, 36]

    for idx in spine_indices:
        # Find the highest probability point for this specific spine index
        x_idx = np.argmax(simcc_x[idx])
        y_idx = np.argmax(simcc_y[idx])
        
        # Convert back to video pixel scale
        x = int((x_idx / len(simcc_x[idx])) * w)
        y = int((y_idx / len(simcc_y[idx])) * h)
        spine_points.append((x, y))

        # Draw the red vertebrae node
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Connect the spine points with a thick green line
    for i in range(1, len(spine_points)):
        cv2.line(frame, spine_points[i-1], spine_points[i], (0, 255, 0), 4)

    # 5. Show the video
    cv2.imshow("Spine Tracker Debug", frame)
    # 5. Show the video
    cv2.imshow("Spine Tracker", frame)
    
    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()