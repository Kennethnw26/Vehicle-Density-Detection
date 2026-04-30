from ultralytics import YOLO
import cv2
import numpy as np
import math

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/traffic4.mp4")

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
CONF_THRESHOLD = 0.4

# Lane state
lanes = []
current_lane = []
drawing_mode = True

# Tracker state
next_id = 0
tracked_objects = {}   # id -> {"pos": (cx, cy), "cls": str, "conf": float, "age": int}
MAX_LOST_FRAMES = 5    # frames before dropping a lost track
DIST_THRESHOLD = 60

DENSITY_COLORS = {
    "LOW":    (0, 200, 0),
    "MEDIUM": (0, 165, 255),
    "HIGH":   (0, 0, 220),
}


def classify_density(count):
    if count <= 3:
        return "LOW"
    elif count <= 7:
        return "MEDIUM"
    return "HIGH"



def mouse_callback(event, x, y, flags, param):
    global current_lane, lanes, drawing_mode

    if not drawing_mode:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        current_lane.append((x, y))

        if len(current_lane) == 4:
            lanes.append(np.array(current_lane, dtype=np.int32))
            print(f"Lane {len(lanes)} added.")
            current_lane = []


def draw_lanes(frame, lane_counts):
    for i, lane in enumerate(lanes):
        density = classify_density(lane_counts[i])
        color = DENSITY_COLORS[density]

        overlay = frame.copy()
        cv2.fillPoly(overlay, [lane], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [lane], True, color, 2)

        cx = int(np.mean(lane[:, 0]))
        cy = int(np.mean(lane[:, 1]))
        cv2.putText(frame, f"L{i+1}: {lane_counts[i]} ({density})",
                    (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_partial_lane(frame):
    if not drawing_mode or not current_lane:
        return
    for pt in current_lane:
        cv2.circle(frame, pt, 6, (0, 255, 255), -1)
    for i in range(1, len(current_lane)):
        cv2.line(frame, current_lane[i - 1], current_lane[i], (0, 255, 255), 1)


def draw_hud(frame, total_vehicles):
    h, w = frame.shape[:2]

    if drawing_mode:
        msg = f"DRAW MODE | Lanes: {len(lanes)} | Click 4 pts/lane | [S] start detection | [R] reset | [Q/ESC] quit"
        color = (0, 255, 255)
    else:
        msg = f"DETECT MODE | Lanes: {len(lanes)} | Vehicles: {total_vehicles} | [R] reset | [Q/ESC] quit"
        color = (0, 255, 0)

    cv2.rectangle(frame, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.putText(frame, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)


def match_detections_to_tracks(detections, tracked_objects, dist_threshold):
    """Simple nearest-neighbour matching with age-based pruning."""
    global next_id

    updated = {}

    for det in detections:
        cx, cy, x1, y1, x2, y2, cls_name, conf = det

        best_id = None
        best_dist = dist_threshold  # shrinks as we find closer matches
        for obj_id, info in tracked_objects.items():
            px, py = info["pos"]
            d = math.hypot(cx - px, cy - py)
            if d < best_dist:
                best_dist = d
                best_id = obj_id

        if best_id is None:
            best_id = next_id
            next_id += 1

        updated[best_id] = {
            "pos": (cx, cy),
            "box": (x1, y1, x2, y2),
            "cls": cls_name,
            "conf": conf,
            "age": 0,
        }

    # Keep lost tracks alive for a few frames
    for obj_id, info in tracked_objects.items():
        if obj_id not in updated:
            if info["age"] < MAX_LOST_FRAMES:
                info["age"] += 1
                updated[obj_id] = info

    return updated


cv2.namedWindow("Vehicle Density Detection")
cv2.setMouseCallback("Vehicle Density Detection", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        # Loop video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD or cls not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = y2  # bottom center (more stable for lane assignment)
        cls_name = VEHICLE_CLASSES[cls]

        detections.append((cx, cy, x1, y1, x2, y2, cls_name, conf))

    tracked_objects = match_detections_to_tracks(detections, tracked_objects, DIST_THRESHOLD)

    # Draw tracked vehicles
    for obj_id, info in tracked_objects.items():
        if "box" not in info:
            continue
        x1, y1, x2, y2 = info["box"]
        cls_name = info["cls"]
        conf = info["conf"]
        label = f"{cls_name} #{obj_id} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

    # Lane vehicle counts
    lane_counts = [0] * len(lanes)
    for info in tracked_objects.values():
        cx, cy = info["pos"]
        for i, lane in enumerate(lanes):
            if cv2.pointPolygonTest(lane, (float(cx), float(cy)), False) >= 0:
                lane_counts[i] += 1

    draw_lanes(frame, lane_counts)
    draw_partial_lane(frame)
    draw_hud(frame, len([i for i in tracked_objects.values() if i["age"] == 0]))

    cv2.imshow("Vehicle Density Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        drawing_mode = False
        print(f"Detection started. {len(lanes)} lane(s) active.")

    elif key == ord('r'):
        lanes = []
        current_lane = []
        tracked_objects = {}
        next_id = 0
        drawing_mode = True
        print("Reset.")

    elif key in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
