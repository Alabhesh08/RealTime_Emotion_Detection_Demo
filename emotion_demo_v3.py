import cv2
import time
from collections import defaultdict
from datetime import datetime
from deepface import DeepFace

# ---------------------------
# SETTINGS
# ---------------------------

CAMERA_INDEX = 0
ANALYZE_EVERY_N_FRAMES = 15

# ---------------------------
# INIT
# ---------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

current_emotion = "Detecting..."
current_confidence = 0.0
emotion_scores = {}

emotion_counts = defaultdict(int)

frame_count = 0

fps = 0
prev_time = time.time()

print("Press ESC to quit")
print("Press S to save screenshot")

# ---------------------------
# MAIN LOOP
# ---------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    try:

        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:

            # result = DeepFace.analyze(
            #     frame,
            #     actions=["emotion"],
            #     enforce_detection=False,
            #     silent=True
            # )

            result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    silent=True
                    )

            result = result[0]

            emotion_scores = result["emotion"]

            current_emotion = result["dominant_emotion"]

            current_confidence = emotion_scores[current_emotion]

            emotion_counts[current_emotion] += 1

    except Exception as e:
        print(e)

    # ---------------------------
    # FACE DETECTION
    # ---------------------------

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        label = (
            f"{current_emotion.upper()} "
            f"({current_confidence:.1f}%)"
        )

        # black background for text
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        cv2.rectangle(
            frame,
            (x, y - 35),
            (x + tw + 10, y),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ---------------------------
    # SIDE PANEL
    # ---------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (250, 600),
        (0, 0, 0),
        -1
    )

    alpha = 0.45

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )

    y = 30

    def draw(text, y_pos, scale=0.6):
        cv2.putText(
            frame,
            text,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            2
        )

    draw("REAL-TIME EMOTION DETECTION", y)
    y += 35

    draw(f"Current: {current_emotion}", y)
    y += 30

    draw(f"Confidence: {current_confidence:.1f}%", y)
    y += 30

    draw(f"FPS: {fps:.1f}", y)
    y += 40

    draw("TOP EMOTIONS", y)
    y += 30

    if emotion_scores:

        top_emotions = sorted(
            emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        for emotion, score in top_emotions:

            draw(
                f"{emotion:<10} {score:.1f}%",
                y,
                0.55
            )

            y += 25

    y += 20

    draw("SESSION STATS", y)
    y += 30

    total = sum(emotion_counts.values())

    if total > 0:

        session_top = max(
            emotion_counts,
            key=emotion_counts.get
        )

        draw(
            f"Session Mood: {session_top}",
            y
        )

        y += 30

        for emotion, count in sorted(
            emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:

            pct = 100 * count / total

            draw(
                f"{emotion}: {pct:.1f}%",
                y,
                0.55
            )

            y += 25

    # ---------------------------
    # DISPLAY
    # ---------------------------

    cv2.imshow(
        "Real-Time Affective State Monitoring",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

    # S -> save screenshot
    elif key == ord('s'):

        filename = (
            f"{current_emotion}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"Saved screenshot: {filename}"
        )

# ---------------------------
# CLEANUP
# ---------------------------

cap.release()
cv2.destroyAllWindows()