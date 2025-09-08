import cv2
import numpy as np
import os
from pathlib import Path

class Model:
    def __init__(self,
                 model_path: Path = Path("./model"),
                 image_path: Path = Path("./images"),
                 face_size=(200, 200),
                 confidence_threshold: float = 80):

        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        self.image_path = image_path
        self.model_path = model_path
        self.face_size = face_size
        self.confidence_threshold = confidence_threshold
        self.label_map = {}

    def train(self):
        usernames = [d.name for d in self.image_path.iterdir() if d.is_dir()]
        training_data, labels = [], []
        self.label_map = {}
        label_counter = 0

        for username in usernames:
            user_dir = self.image_path / username
            for file in os.listdir(user_dir):
                image = cv2.imread(str(user_dir / file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = cv2.resize(image, self.face_size)
                training_data.append(image)
                labels.append(label_counter)

            self.label_map[label_counter] = username
            label_counter += 1

        if not training_data:
            print("No training images found.")
            return

        labels = np.asarray(labels, dtype=np.int32)
        self.face_recognizer.train(training_data, labels)

        self.model_path.mkdir(parents=True, exist_ok=True)
        self.face_recognizer.save(str(self.model_path / "face_recognizer.xml"))
        np.save(self.model_path / "labels.npy", self.label_map)

        print("Model trained successfully!")

    def load_model(self):
        model_file = self.model_path / "face_recognizer.xml"
        label_file = self.model_path / "labels.npy"

        if not model_file.exists() or not label_file.exists():
            print("No model found.")
            return False

        self.face_recognizer.read(str(model_file))
        self.label_map = np.load(label_file, allow_pickle=True).item()
        return True

    def predict(self, face):
        if face is None:
            return "Unknown", 0

        l, conf_val = self.face_recognizer.predict(face)
        confidence = max(0, 100 * (1 - conf_val / self.confidence_threshold))

        if conf_val <= self.confidence_threshold:
            return self.label_map.get(l, "Unknown"), confidence

        return "Unknown", 0
