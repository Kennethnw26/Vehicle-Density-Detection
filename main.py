from ultralytics import YOLO
import cv2
import numpy as np
import math

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/traffic4.mp4")

vehicle_classes = [2, 3, 5, 7]
CONF_THRESHOLD = 0.5

# Multi Lane 
lanes = []
current_lane = []
drawing_mode = True  # control whether user is drawing lanes

# Tracking
next_id = 0
tracked_objects = {}  # id: (cx, cy)

DIST_THRESHOLD = 50


def classify_density(count):
    if count <= 3:
        return "LOW"
    elif count <= 7:
        return "MEDIUM"
    else:
        return "HIGH"


# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global current_lane, lanes, drawing_mode

    if not drawing_mode:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        current_lane.append((x, y))
        print(f"Point: {(x, y)}")

        if len(current_lane) == 4:
            lanes.append(np.array(current_lane))
            print("Lane added!")
            current_lane = []

cv2.namedWindow("Multi-Lane Detection")
cv2.setMouseCallback("Multi-Lane Detection", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    detections = []

    # -------- DETECTION --------
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = (x1 + x2) // 2
            cy = y2  # bottom center

            detections.append((cx, cy, x1, y1, x2, y2))

    # -------- TRACKING --------
    new_tracked = {}

    for det in detections:
        cx, cy, x1, y1, x2, y2 = det

        matched_id = None

        for obj_id, (px, py) in tracked_objects.items():
            dist = math.hypot(cx - px, cy - py)

            if dist < DIST_THRESHOLD:
                matched_id = obj_id
                break

        if matched_id is None:
            matched_id = next_id
            next_id += 1

        new_tracked[matched_id] = (cx, cy)

        # Draw bounding box + ID
        label = f"ID {matched_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    tracked_objects = new_tracked

    # -------- MULTI-LANE COUNT --------
    lane_counts = [0 for _ in lanes]

    for obj_id, (cx, cy) in tracked_objects.items():
        for i, lane in enumerate(lanes):
            if cv2.pointPolygonTest(lane, (cx, cy), False) >= 0:
                lane_counts[i] += 1

    # -------- DRAW LANES --------
    for i, lane in enumerate(lanes):
        cv2.polylines(frame, [lane], True, (255,0,0), 2)

        density = classify_density(lane_counts[i])

        cv2.putText(frame,
                    f"Lane {i+1}: {lane_counts[i]} ({density})",
                    (20, 60 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2)

    # -------- DRAW CURRENT LANE (while clicking) --------
    if drawing_mode and len(current_lane) > 0:
        for point in current_lane:
            cv2.circle(frame, point, 5, (0,255,255), -1)

    # -------- MODE DISPLAY --------
    if drawing_mode:
        cv2.putText(frame,
                    "DRAW MODE: Click 4 points per lane, press 's' to start",
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2)
    else:
        cv2.putText(frame,
                    "DETECTION MODE (press 'r' to reset)",
                    (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2)

    cv2.imshow("Multi-Lane Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Lock lanes
    if key == ord('s'):
        drawing_mode = False
        print("Lanes locked. Detection started.")

    # Reset everything
    if key == ord('r'):
        lanes = []
        current_lane = []
        tracked_objects = {}
        drawing_mode = True
        print("Reset everything")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()