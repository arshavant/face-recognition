import cv2
from config import IMAGE_DIR, FACE_SIZE
from model import train_model, load_model, detect_face, predict_face

camera = cv2.VideoCapture(0)


def capture_images(username: str, max_images: int = 500):
    user_dir = IMAGE_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    count = 1
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while count <= max_images:
        ret, frame = camera.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        cropped_face = gray[y:y + h, x:x + w]
        resized_face = cv2.resize(cropped_face, FACE_SIZE)

        cv2.imshow("Capture", resized_face)
        cv2.imwrite(str(user_dir / f"{count}.jpg"), resized_face)
        count += 1

        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()
    print(f"Collected {count-1} images for {username}")


def recognize_live(label_map):
    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        image, face, pos = detect_face(frame)
        name, confidence = predict_face(face, label_map)
        cv2.putText(image, f'{name} ({confidence}%)', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Recognition", image)

        if cv2.waitKey(1) & 0xFF == 13:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice = input("Capture images? (y/n): ").lower()
    if choice in ("y", "yes"):
        username = input("Enter username: ")
        capture_images(username)

    choice = input("Train model? (y/n): ").lower()
    if choice in ("y", "yes"):
        label_map = train_model()
    else:
        if load_model():
            print("Model loaded successfully.")
            label_map = train_model()
        else:
            print("No model found. Capture and train first.")
            exit()

    choice = input("Recognize face? (y/n): ").lower()
    if choice in ("y", "yes"):
        recognize_live(label_map)
