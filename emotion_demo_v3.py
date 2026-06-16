import cv2
import time
from collections import defaultdict, deque, Counter
from datetime import datetime
from deepface import DeepFace

# =====================================================
# SETTINGS
# =====================================================

CAMERA_INDEX = 0

ANALYZE_EVERY_N_FRAMES =10

SMOOTHING_WINDOW = 5

CONFIDENCE_THRESHOLD = 30

# =====================================================
# INIT
# =====================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

frame_count = 0

emotion_scores = {}

current_emotion = "Detecting"
current_confidence = 0

stable_emotion = "Detecting"

emotion_history = deque(maxlen=SMOOTHING_WINDOW)

emotion_counts = defaultdict(int)

fps = 0
prev_time = time.time()

print("ESC = Exit")
print("S = Save Screenshot")

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # -------------------------
    # FPS
    # -------------------------

    current_time = time.time()

    fps = 1 / max(
        current_time - prev_time,
        1e-6
    )

    prev_time = current_time

    # -------------------------
    # EMOTION ANALYSIS
    # -------------------------

    try:

        if frame_count % ANALYZE_EVERY_N_FRAMES == 0:

            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            # result = DeepFace.analyze(
            #     frame,
            #     actions=["emotion"],
            #     detector_backend="opencv",
            #     enforce_detection=False,
            #     silent=True
            # )

            result = result[0]

            emotion_scores = result["emotion"]

            scores = result["emotion"]

            scores.pop("fear", None)
            scores.pop("disgust", None)

            current_emotion = max(
                scores,
                key=scores.get
            )

            current_confidence = scores[current_emotion]

            current_confidence = emotion_scores[current_emotion]

            # confidence filtering

            if current_confidence >= CONFIDENCE_THRESHOLD:

                emotion_history.append(
                    current_emotion
                )

                stable_emotion = Counter(
                    emotion_history
                ).most_common(1)[0][0]

                emotion_counts[
                    stable_emotion
                ] += 1

            else:

                stable_emotion = "Uncertain"

    except Exception as e:
        print("Analysis error:", e)

    # -------------------------
    # FACE DETECTION
    # -------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

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
            f"{stable_emotion.upper()} "
            f"({current_confidence:.1f}%)"
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            2
        )

        cv2.rectangle(
            frame,
            (x, y - 40),
            (x + tw + 15, y),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x + 5, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

    # -------------------------
    # LEFT INFO PANEL
    # -------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (200, 500),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame
    )

    y = 30

    def draw(text, ypos, scale=0.6):

        cv2.putText(
            frame,
            text,
            (10, ypos),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            2
        )

    draw(
        "REAL-TIME EMOTION DETECTION",
        y,
        0.7
    )

    y += 40

    draw(
        f"Current : {current_emotion}",
        y
    )

    y += 30

    draw(
        f"Stable  : {stable_emotion}",
        y
    )

    y += 30

    draw(
        f"Confidence : {current_confidence:.1f}%",
        y
    )

    y += 30

    draw(
        f"FPS : {fps:.1f}",
        y
    )

    y += 40

    draw("TOP EMOTIONS", y)

    y += 30

    if emotion_scores:

        top5 = sorted(
            emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        for emotion, score in top5:

            draw(
                f"{emotion:<10} {score:.1f}%",
                y,
                0.55
            )

            y += 25

    y += 20

    draw("SESSION SUMMARY", y)

    y += 30

    total = sum(
        emotion_counts.values()
    )

    if total > 0:

        session_mood = max(
            emotion_counts,
            key=emotion_counts.get
        )

        draw(
            f"Dominant: {session_mood}",
            y
        )

        y += 30

        for emotion, count in sorted(
            emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            pct = (
                count / total
            ) * 100

            draw(
                f"{emotion}: {pct:.1f}%",
                y,
                0.55
            )

            y += 25

    # -------------------------
    # DISPLAY
    # -------------------------

    cv2.imshow(
        "Real-Time Affective State Monitoring",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ESC

    if key == 27:
        break

    # Screenshot

    elif key == ord('s'):

        filename = (
            f"{stable_emotion}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        print(
            f"Saved: {filename}"
        )

# =====================================================
# CLEANUP
# =====================================================

cap.release()
cv2.destroyAllWindows()

