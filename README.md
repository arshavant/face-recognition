# face-recognition

**Unlock the power of Python + OpenCV in just a single command.**

---

## Snapshot

A minimalist yet powerful tool for face detection and recognition. Built with Python and OpenCV, this project lets you detect faces using a pre-trained Haar cascade and match them with recognition logic wrapped neatly in `model.py`.

---

## Quick Start

```bash
git clone https://github.com/arshavant/face-recognition.git
cd face-recognition
pip install opencv-python opencv-contrib-python
python main.py
```

*That’s it — your face-recognition journey begins.*

---

## What’s Inside

| File                                  | Description                                           |
| ------------------------------------- | ----------------------------------------------------- |
| `main.py`                             | The heart of the app: handles detection + recognition |
| `model.py`                            | Face recognition logic—plug in your own model here!   |
| `config.py`                           | Optional: store settings like thresholds/data paths   |
| `haarcascade_frontalface_default.xml` | Pre-trained detector from OpenCV                      |

---

## How It Works

1. **Detect** faces in the input image using the Haar cascade.
2. **Recognize** using the logic in `model.py` (e.g., feature matching, classification).
3. **Optional**: Tune detection or recognition settings via `config.py`.

---

## Why You'll Love It

* **Light & Clean**: Just the essentials—no extra noise.
* **Modular**: Flexible architecture, ready for feature expansion (think deep learning, video, UI).
* **Learning-Focused**: Perfect for students and devs getting hands-on with OpenCV and Python.


