import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

current_emotion = "Detecting..."
current_confidence = 0
emotion_scores = {}

frame_count = 0

print("Press ESC to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    try:
        # Analyze every 15 frames for speed
        if frame_count % 15 == 0:

            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )

            result = result[0]

            emotion_scores = result["emotion"]

            current_emotion = result["dominant_emotion"]

            current_confidence = emotion_scores[current_emotion]

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
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
                2
            )

            cv2.putText(
                frame,
                f"{current_emotion} ({current_confidence:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Emotion panel
        y_offset = 30

        cv2.putText(
            frame,
            "Emotion Scores",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        y_offset += 30

        for emotion, score in sorted(
            emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:

            text = f"{emotion}: {score:.1f}%"

            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

            y_offset += 25

    except Exception as e:
        print(e)

    cv2.imshow(
        "Real-Time Emotion Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()