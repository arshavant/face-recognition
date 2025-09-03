import cv2
from capture import capture_images
from model import train_all_users, load_all_users_model, predict_face, face_cascade

def recognize_all_live(window_name="Live Recognition"):
    if not load_all_users_model():
        return

    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cropped = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            name, confidence = predict_face(cropped)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}, {confidence}%", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == 13:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        print("\nMenu:")
        print(" [c] Capture Images")
        print(" [t] Train Model")
        print(" [l] Live Recognition")
        print(" [q] Quit")
        choice = input("Choose an option: ").lower()

        if choice in ("c", "capture"):
            username = input("Enter username: ").lower()
            capture_images(username)
        elif choice in ("t", "train"):
            train_all_users()
        elif choice in ("l", "live"):
            recognize_all_live()
        elif choice in ("q", "quit"):
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")
