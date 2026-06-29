import cv2

from src.detector import Detector

detector = Detector()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame, people = detector.detect(frame)

    print("-" * 50)
    print(f"People detected: {len(people)}")

for person in people:
    print(
        f"ID: {person.track_id} | "
        f"Confidence: {person.confidence:.2f} | "
        f"Center: {person.center}"
    )

    cv2.imshow("Detector Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()