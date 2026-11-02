import cv2
import time

cap = cv2.VideoCapture(0)
print("Running OpenPose Simulation... Notice the frame rate drop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Artificially simulate the heavy CPU processing time of OpenPose
    time.sleep(0.8) 
    
    cv2.putText(frame, "OpenPose FPS: 1.2 (Very Slow - CPU Bottleneck)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("OpenPose Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()