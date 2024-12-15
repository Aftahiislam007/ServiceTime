import cv2
import numpy as np
from collections import defaultdict

video_path = "resources/fringestorez.mp4"
cap = cv2.VideoCapture(video_path)

fringe_background = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

time_tracker = defaultdict(list)
customer_id = 0

customer_positions = {}
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fringe_background.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        centroid = (x + w // 2, y + h // 2)

        matched = False
        for cid, pos in customer_positions.items():
            if np.linalg.norm(np.array(pos) - np.array(centroid)) < 50:
                customer_positions[cid] = centroid
                matched = True
                break

        if not matched:
            customer_id += 1
            customer_positions[customer_id] = centroid
            time_tracker[customer_id].append(frame_count / fps) 

    to_remove = []
    for cid, pos in customer_positions.items():
        if not any(cv2.pointPolygonTest(contour, pos, False) >= 0 for contour in contours):
            to_remove.append(cid)

    for cid in to_remove:
        time_tracker[cid].append(frame_count / fps)
        del customer_positions[cid]

cap.release()

service_times = [times[1] - times[0] for times in time_tracker.values() if len(times) == 2]

if service_times:
    average_service_time = sum(service_times) / len(service_times)
    print(f"Average service time: {average_service_time:.2f} seconds")
else:
    print("No customers detected or tracked successfully.")
