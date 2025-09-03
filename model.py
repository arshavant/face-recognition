import cv2
import numpy as np
import os
from config import IMAGE_PATH, MODEL_PATH, CASCADE_PATH, FACE_SIZE, CONFIDENCE_THRESHOLD

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
label_map = {}

def train_all_users():
    global label_map
    usernames = [d.name for d in IMAGE_PATH.iterdir() if d.is_dir()]

    training_data, labels = [], []
    label_map = {}
    label_counter = 0

    for username in usernames:
        user_dir = IMAGE_PATH / username
        for file in os.listdir(user_dir):
            img = cv2.imread(str(user_dir / file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, FACE_SIZE)
            training_data.append(img)
            labels.append(label_counter)

        label_map[label_counter] = username
        label_counter += 1

    if not training_data:
        print("No training images found!")
        return

    labels = np.array(labels, dtype=np.int32)
    face_recognizer.train(training_data, labels)

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    face_recognizer.save(str(MODEL_PATH / "all_users.xml"))
    np.save(MODEL_PATH / "label_map.npy", label_map)
    print("Training complete! Model saved as 'all_users.xml'.")


def load_all_users_model():
    global label_map
    model_file = MODEL_PATH / "all_users.xml"
    map_file = MODEL_PATH / "label_map.npy"

    if not model_file.exists() or not map_file.exists():
        print("Model not found. Train first!")
        return False

    face_recognizer.read(str(model_file))
    label_map = np.load(map_file, allow_pickle=True).item()
    return True


def predict_face(face):
    if face is None:
        return "Unknown", 0

    label, conf_val = face_recognizer.predict(face)
    confidence = max(0, int(100 * (1 - conf_val / CONFIDENCE_THRESHOLD)))

    if conf_val < CONFIDENCE_THRESHOLD:
        return label_map[label], confidence
    return "Unknown", confidence
