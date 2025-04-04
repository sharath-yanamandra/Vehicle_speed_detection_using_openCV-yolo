# Vehicle_speed_detection_using_openCV-yolo
# Speed Detection Using OpenCV

This project provides a simple Python script to calculate the speed of a moving object in a video using OpenCV.

## 📹 Overview

The script reads a video file, detects moving objects in the frame using background subtraction, calculates the distance moved in pixels, and estimates the speed based on real-world distance calibration.

---

## 🚀 Features

- Object detection using background subtraction (`cv2.createBackgroundSubtractorMOG2`)
- Centroid tracking and movement calculation
- Real-world distance calibration using a known reference
- Speed estimation in km/h

---

## 🧰 Requirements

- Python 3.x
- OpenCV (`cv2`)
- Numpy

Install dependencies via pip:

```bash
# pip install opencv-python numpy

## 📁 How to Use
Replace the video source: Update the following line to your own video file:

python
Copy
Edit
cap = cv2.VideoCapture("your_video.mp4")
Adjust calibration values: Modify real_world_distance_meters and pixels_between_lines to match your scene calibration.

Run the script:

bash
Copy
Edit
python speed.py
Press Esc to quit the video window.
