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
```
## 📁 How to Use
Replace the video source: Update the following line to your own video file:

```
cap = cv2.VideoCapture("your_video.mp4")
```
Adjust calibration values: Modify real_world_distance_meters and pixels_between_lines to match your scene calibration.

Run the script:
```
python speed.py
```
Press Esc to quit the video window.

## ⚙️ Parameters
```
Parameter -->	Description
min_contour_width -->	Minimum width to consider an object
min_contour_height -->	Minimum height to consider an object
real_world_distance_meters -->	Real-world distance between lines in meters
pixels_between_lines	--> Pixel distance between two reference lines
frame_rate -->	Frame rate of the video (used for speed)
```

## 📦 Output
Displays the video with detected objects.

Shows calculated speed on the video frame.
```
(Optional) Save output by modifying cv2.VideoWriter.
```

## 📌 Notes
Make sure the camera is fixed and not moving.

The object should cross two lines to calculate speed accurately.

You may enhance detection by applying object tracking (e.g., SORT, Deep SORT).

## 📄 License
This project is open-source and available under the MIT License.

```
Let me know if you'd like me to also help you generate a `requirements.txt`, add object tracking, or convert this to a Streamlit or Flask web app!
```







