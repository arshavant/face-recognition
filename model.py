import cv2
import numpy as np
from config import IMAGE_DIR, MODEL_DIR, FACE_SIZE, CONFIDENCE_THRESHOLD, CASCADE_PATH

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def train_model():
    images, labels, label_map = [], [], {}
    label_counter = 0

    for user_dir in IMAGE_DIR.iterdir():
        if not user_dir.is_dir():
            continue
        label_map[label_counter] = user_dir.name
        for file in user_dir.iterdir():
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, FACE_SIZE)
            images.append(img)
            labels.append(label_counter)
        label_counter += 1

    if not images:
        print("No images found to train.")
        return None

    labels = np.array(labels, dtype=np.int32)
    face_recognizer.train(np.array(images, dtype=np.uint8), labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    face_recognizer.save(str(MODEL_DIR / "face_recognizer.xml"))
    print(f"Model trained and saved. Users: {list(label_map.values())}")
    return label_map


def load_model():
    model_path = MODEL_DIR / "face_recognizer.xml"
    if model_path.exists():
        face_recognizer.read(str(model_path))
        return True
    return False


def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return frame, None, (0, 0)

    x, y, w, h = faces[0]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped_face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
    return frame, cropped_face, (x, y)


def predict_face(face, label_map):
    if face is None:
        return "unknown", 0
    label, distance = face_recognizer.predict(face)
    confidence = max(0, int(100 * (1 - distance / CONFIDENCE_THRESHOLD)))
    if distance < CONFIDENCE_THRESHOLD and label in label_map:
        return label_map[label], confidence
    return "unknown", confidence
