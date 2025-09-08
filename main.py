from capture import Capture
from model import Model
from live_detection import LiveDetection

def main():
    while True:
        print("\nOptions")
        print("[c] Capture Images")
        print("[t] Train Model")
        print("[l] Live Detection")
        print("[q] Quit")

        choice = input("Choose an option: ").lower()

        if choice in ("c", "capture images"):
            username = input("Enter username: ")
            Capture().run(username=username, max_images=100)

        elif choice in ("t", "train model"):
            Model().train()

        elif choice in ("l", "live detection"):
            LiveDetection().recognize()

        elif choice in ("q", "quit"):
            print("Quitting...")
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
