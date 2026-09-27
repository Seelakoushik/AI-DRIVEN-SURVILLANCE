 from ultralytics import YOLO
import cv2
import numpy as np
import time
import winsound

# Load YOLO Model
model = YOLO("yolov8n.pt")

# Camera
cap = cv2.VideoCapture(0)

# Restricted Area
restricted_zone = np.array([
    [100, 100],
    [500, 100],
    [500, 400],
    [100, 400]
])

# Dangerous Objects
dangerous_objects = ["knife", "scissors"]

# Loitering Settings
person_positions = {}
loiter_threshold = 15

# Alarm Cooldown
last_alarm_time = 0
alarm_cooldown = 3

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera Error")
        break

    results = model.track(
        frame,
        persist=True,
        verbose=False
    )

    annotated = frame.copy()
    person_count = 0

    # Threat Flags
    weapon_detected = False
    intrusion_detected = False
    loitering_detected = False
    crowd_detected = False

    # Draw Restricted Zone
    cv2.polylines(
        annotated,
        [restricted_zone],
        True,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated,
        "RESTRICTED AREA",
        (120, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            color = (0, 255, 0)

            # PERSON DETECTION
            if label == "person":

                person_count += 1
                color = (255, 0, 0)

                # Intrusion Detection
                inside = cv2.pointPolygonTest(
                    restricted_zone,
                    (cx, cy),
                    False
                )

                if inside >= 0:

                    intrusion_detected = True

                    cv2.putText(
                        annotated,
                        "INTRUSION ALERT",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                # Loitering Detection
                if box.id is not None:

                    person_id = int(box.id[0])

                    if person_id not in person_positions:
                        person_positions[person_id] = time.time()

                    duration = (
                        time.time()
                        - person_positions[person_id]
                    )

                    if duration > loiter_threshold:

                        loitering_detected = True

                        cv2.putText(
                            annotated,
                            f"LOITERING ID:{person_id}",
                            (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

            # WEAPON DETECTION
            if label in dangerous_objects:

                weapon_detected = True
                color = (0, 0, 255)

                cv2.putText(
                    annotated,
                    "WEAPON DETECTED",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

            # Draw Bounding Box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Crowd Density Analysis
    if person_count > 10:

        crowd_detected = True

        cv2.putText(
            annotated,
            "HIGH CROWD DENSITY",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 165, 255),
            3
        )

    # People Count
    cv2.putText(
        annotated,
        f"People Count: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ALARM SYSTEM
    current_time = time.time()

    if current_time - last_alarm_time > alarm_cooldown:

        if weapon_detected:

            winsound.Beep(2500, 1000)
            winsound.Beep(2500, 1000)

            cv2.imwrite(
                f"weapon_{int(time.time())}.jpg",
                annotated
            )

            last_alarm_time = current_time

        elif intrusion_detected:

            winsound.Beep(1800, 500)
            winsound.Beep(1800, 500)

            cv2.imwrite(
                f"intrusion_{int(time.time())}.jpg",
                annotated
            )

            last_alarm_time = current_time

        elif loitering_detected:

            winsound.Beep(1200, 700)

            cv2.imwrite(
                f"loitering_{int(time.time())}.jpg",
                annotated
            )

            last_alarm_time = current_time

        elif crowd_detected:

            winsound.Beep(900, 400)
            winsound.Beep(900, 400)

            cv2.imwrite(
                f"crowd_{int(time.time())}.jpg",
                annotated
            )

            last_alarm_time = current_time

    # General Threat Message
    if (
        weapon_detected
        or intrusion_detected
        or loitering_detected
        or crowd_detected
    ):

        cv2.putText(
            annotated,
            "THREAT DETECTED!",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "AI Driven Surveillance System",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
