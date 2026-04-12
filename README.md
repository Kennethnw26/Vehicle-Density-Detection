# Vehicle-Density-Detection
# 🚦 Vehicle Density Detection System

## Overview

This project is a **real-time vehicle density detection system** built using **Computer Vision and AI**. It detects vehicles from traffic video footage, tracks them across frames, and calculates **lane-based traffic density**.

The system allows users to manually define lanes using an interactive interface and provides real-time feedback on traffic conditions.

---

## Features

* Real-time vehicle detection using YOLOv8
* Confidence-based filtering for accurate detection
* Multi-lane support with polygon (trapezium) ROI
* Interactive lane selection (click-based)
* Vehicle counting per lane
* Density classification (LOW / MEDIUM / HIGH)
* Basic vehicle tracking using centroid matching
* Reset and dynamic configuration support

---

## How It Works

1. **Video Input**
   A traffic video is processed frame by frame.

2. **Vehicle Detection**
   YOLOv8 detects objects and filters only vehicle classes (car, motorcycle, bus, truck).

3. **Tracking**
   Each detected vehicle is assigned an ID to prevent duplicate counting.

4. **Lane Definition (ROI)**
   Users define lanes by clicking 4 points per lane (forming a trapezium).

5. **Lane Assignment**
   Vehicles are assigned to lanes using polygon detection.

6. **Density Calculation**
   Each lane is classified based on vehicle count:

   * LOW
   * MEDIUM
   * HIGH

---

## Controls

| Key   | Action                                 |
| ----- | -------------------------------------- |
| Click | Select lane points (4 points per lane) |
| `s`   | Lock lanes and start detection         |
| `r`   | Reset all lanes and tracking           |
| `ESC` | Exit program                           |

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Ultralytics YOLOv8

---

## 📂 Project Structure

```
Vehicle-Density-Detection/
│
├── main.py
├── requirements.txt
├── README.md
├── yolov8n.pt
├── videos/          # (Not included in repo)
```

---

## 📥 Video Files (Important)

Due to GitHub's file size limitations, video files are **not included in this repository**.

### 📦 Download Videos

The videos are provided as a **videos.zip file**.

### 📁 Setup Instructions

1. Download the ZIP file
2. Extract it
3. Place the videos inside a folder named:

```
videos/
```

So your structure becomes:

```
Vehicle-Density-Detection/
├── main.py
├── videos/
│   ├── traffic.mp4
│   ├── traffic2.mp4
│   └── ...
```

---

## Limitations

* Tracking is basic (may lose IDs in complex scenes)
* Requires manual lane selection
* Performance depends on video quality and stability

---

## Future Improvements

* Advanced tracking (DeepSORT)
* Automatic lane detection
* Traffic light decision system
* Save/load lane configurations
* Perspective transformation (bird's-eye view)


## Notes

This project was developed as part of a learning journey in **AI and Computer Vision**, focusing on understanding real-time detection systems and spatial analysis. This also related to my proposal in Research Methods for Computing Technology, where I proposed as "Development of Deep Learning-Based Adaptive Traffic Signal Systems Using Weather and Real-Time Vehicle Density Detection" and this is one step that helps and guides me towards achieving it in case I do want to pursue this into my Final Year Project. I am driven by curiosity and what AI and Computer Vision could do. This project is made with the guide of AI tools so it is not all written by me but I did make some tweaks to suit my objective and goals. 

