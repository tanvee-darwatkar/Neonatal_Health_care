import cv2
import numpy as np
import time
import winsound
from shared_data import dashboard_data

CAMERA_INDEX = 0
FRAME_SIZE = (128, 128)
MOVEMENT_THRESHOLD = 12
MAX_STILL_TIME = 15

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

prev_frame = None
still_start_time = None
alert_triggered = False

print("👶 Smart Neonatal AI Monitoring Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera frame not received")
        break

    h, w, _ = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, FRAME_SIZE)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    movement_detected = False
    movement_score = 0

    if prev_frame is not None:
        diff = cv2.absdiff(gray, prev_frame)
        movement_score = float(np.mean(diff))
        movement_detected = movement_score > MOVEMENT_THRESHOLD

    prev_frame = gray.copy()

    if not movement_detected:
        if still_start_time is None:
            still_start_time = time.time()
    else:
        still_start_time = None
        alert_triggered = False

    still_duration = int(time.time() - still_start_time) if still_start_time else 0

    if movement_detected:
        status = "SAFE"
    elif still_duration < MAX_STILL_TIME:
        status = "STILL (Monitoring)"
    else:
        status = "ALERT: Prolonged Stillness"
        if not alert_triggered:
            winsound.Beep(1000, 800)
            alert_triggered = True

    # ✅ UPDATE SHARED DATA (THIS IS WHAT FASTAPI READS)
    dashboard_data["status"] = status
    dashboard_data["stillTime"] = int(still_duration)
    dashboard_data["motion"] = float(movement_score)

    if status == "ALERT: Prolonged Stillness":
        dashboard_data["confidence"] = 98
        dashboard_data["alertActive"] = True
    else:
        dashboard_data["confidence"] = 90 if movement_detected else 85
        dashboard_data["alertActive"] = False

    cv2.putText(frame, f"STATUS: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Neonatal AI Monitor", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()