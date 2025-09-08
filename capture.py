import cv2
from pathlib import Path

class Capture:
    def __init__(self,
                 model_path: Path = Path("./haarcascade_frontalface_default.xml"),
                 image_path: Path = Path("./images")):

        self.camera = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(str(model_path))
        self.image_path = image_path
        self.count = 0

    def process_image(self, frame, face_size=(200, 200)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            return cv2.resize(gray[y:y + h, x:x + w], face_size)

        return None

    def run(self, username: str, max_images: int = 200, window_title: str = "Capturing Images..."):
        user_dir = self.image_path / username
        user_dir.mkdir(parents=True, exist_ok=True)

        while self.count < max_images:
            ret, frame = self.camera.read()
            if not ret:
                continue

            processed = self.process_image(frame)
            if processed is not None:
                cv2.imshow(window_title, processed)
                cv2.imwrite(str(user_dir / f"{self.count}.jpg"), processed)
                self.count += 1

            if cv2.waitKey(1) & 0xFF == 13:
                break

        self.camera.release()
        cv2.destroyAllWindows()
        print(f"Captured {self.count} images for {username}")
