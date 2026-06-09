"""
This is a simple example of how to use the MediaPipe Hand Landmarker in Python to
detect hand landmarks in a video feed from the webcam. The code captures video frames, 
processes them with the hand landmarker, and visualizes the detected landmarks and 
connections on the video feed. Press the 'Esc' key to exit the video window.

Not to be used in production. Remove when not required.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "model/hand_landmarker.task"

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),      # Thumb
    (0,5),(5,6),(6,7),(7,8),      # Index
    (5,9),(9,10),(10,11),(11,12), # Middle
    (9,13),(13,14),(14,15),(15,16), # Ring
    (13,17),(17,18),(18,19),(19,20), # Pinky
    (0,17)                        # Palm
]

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)

cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    timestamp = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror image
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp += 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        h, w, _ = frame.shape

        for hand in result.hand_landmarks:
            points = []

            # Draw landmarks
            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # Draw connections
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()