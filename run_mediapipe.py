import cv2
import time

cap = cv2.VideoCapture(0)
print("Running CPU Motion Tracker... Real-time performance.")

# Read the first two frames to compare them
ret, frame1 = cap.read()
ret, frame2 = cap.read()

prev_time = 0

while cap.isOpened():
    # Calculate the mathematical difference between frames to detect movement
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Only draw boxes around significant movement (like waving your arms)
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Calculate real-time FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    cv2.putText(frame1, f"Lightweight CPU Tracking FPS: {fps:.1f} (Real-time)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("CPU Real-Time Performance", frame1)

    # Advance the frames
    frame1 = frame2
    ret, frame2 = cap.read()
    if not ret: break

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()