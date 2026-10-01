# AI-Powered Elderly Fall Detection & Healthcare Monitoring System

An end-to-end computer vision and pose-estimation application developed with **MediaPipe Tasks** and **Streamlit** to detect elderly posture, classify activities in real time, and trigger emergency alerts upon fall detection.

---

## 📌 Project Overview

Elderly falls represent a major healthcare risk requiring prompt intervention. This project builds a functional, lightweight, and deployable healthcare monitoring dashboard that tracks human body landmarks (shoulders, hips, knees, feet) and classifies activities into six key states:

1. **Fall** (Emergency: body tipped over or rapid high-angle lean)
2. **Off Balance** (Warning: sudden or acute leaning movement)
3. **Sitting** (Chair sit detection using thigh depth & knee angles)
4. **Standing** (Upright posture alignment)
5. **Walking** (Alternating leg step width, ankle lift, or progressive hip translation)
6. **Normal** (No person detected in the frame)

The application supports multiple input sources: single photo uploads, video files, single webcam snapshots, and live real-time WebRTC streaming.

---

## 🚀 Key Features

- **MediaPipe Pose Landmarker Integration:** Utilizes MediaPipe's lightweight task model (`pose_landmarker.task`) for low-latency body landmark extraction on CPU.
- **Biomechanical Rule-Based Classification:**
  - **Trunk Tilt Calculation:** Evaluates shoulder-to-hip vector angle relative to the vertical axis.
  - **Dynamic Tilt Delta:** Tracks frame-by-frame angular velocity to spot rapid off-balance drops.
  - **Knee Flexion & Thigh Depth:** Uses law of cosines on hip-knee-ankle coordinates and relative $z$-depth coordinate disparities to distinguish between sitting and standing.
  - **Locomotion / Gait Tracking:** Evaluates inter-ankle distance, foot lift differential, and multi-frame hip trajectory vectors.
- **Multi-Modal Input Modes:**
  - 📷 **Photo:** Inspects static images and outputs confidence scores and activity states.
  - 🎥 **Video:** Decodes uploaded video feeds via PyAV in memory, aggregates activity counts, and generates an activity distribution bar chart.
  - 📸 **Camera:** Single snapshot analysis via native Streamlit camera input.
  - 📡 **Live:** Low-latency, full-duplex video streaming powered by `streamlit-webrtc` with background thread landmark detection.
- **Emergency Alert System:** Instant visual indicators and alerts (`EMERGENCY`, `Warning`, `Success`) configured based on user stability.
- **Accessible Healthcare UI:** Custom styling with light and dark mode toggles, color-coded pose landmark skeletons, and Altair analytics charts.

---

## 🛠️ System Architecture & Workflow

```text
[Input Feed: Photo / Video / Live Camera]
                    │
                    ▼
          [Frame Preprocessing]
     (BGR to RGB, Contiguous Array)
                    │
                    ▼
     [MediaPipe Pose Landmarker Tasks]
 (Detects 33 Landmark Keypoints: Shoulders, Hips, Knees, Feet)
                    │
                    ▼
     [Biomechanical Feature Extraction]
  ├── Trunk Tilt Angle (atan2)
  ├── Left & Right Knee Angles (Vector Cosine)
  ├── 3D Depth Disparity (z-axis knee vs. hip)
  └── Multi-frame Hip Trajectory & Step Disparity
                    │
                    ▼
       [Classification & Alert Logic]
  ├── Fall (Tilt > 35°) ──> [EMERGENCY ALERT]
  ├── Off Balance (Tilt > 20° or Rapid Delta) ──> [WARNING]
  ├── Sitting / Standing / Walking
  └── Normal (No Person)
                    │
                    ▼
     [Streamlit Dashboard Output]
  ├── Pose Overlay Skeleton Visualization
  ├── Real-time Confidence / Activity Frequency Charts
  └── Healthcare Status Messages
```

---

## 📂 Project Structure

```text
.
├── app.py                  # Main Streamlit web application & classification pipeline
├── pose_landmarker.task    # MediaPipe pose landmarker model asset
├── logo.png                # Healthcare application logo
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation and assignment overview
```

---

## ⚙️ Installation & Local Setup

### 1. Prerequisites
- Python 3.9 – 3.11 installed on your system.
- Webcam access (for Camera and Live modes).

### 2. Clone the Repository
```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

### 3. Create a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
Create a `requirements.txt` file (if not already present):
```text
streamlit>=1.30.0
mediapipe>=0.10.9
opencv-python-headless>=4.8.0
numpy>=1.24.0
altair>=5.0.0
av>=11.0.0
streamlit-webrtc>=0.47.0
```
Then run:
```bash
pip install -r requirements.txt
```

### 5. Download the Model File
Make sure the MediaPipe model asset is located in the root directory:
- Filename: `pose_landmarker.task`
- Download from Google MediaPipe: [Pose Landmarker (Heavy or Full/Float16 task bundle)](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models).

### 6. Run the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Evaluation & Classification Logic

| Activity State | Biomechanical / Spatial Detection Criteria | Alert Level |
| :--- | :--- | :--- |
| **Fall** | Torso tilt angle $> 35^\circ$ from the vertical axis. | **Critical Emergency** |
| **Off Balance** | Torso tilt between $20^\circ$ and $35^\circ$, or a tilt increase $> 12^\circ$ in subsequent frames with tilt $> 18^\circ$. | **Warning Alert** |
| **Sitting** | Knee bend $< 130^\circ$, thigh span $> 0.05$, or forward-facing thigh-knee $z$-depth differential ($z_{hip} - z_{knee} > 0.08$). | Normal State |
| **Walking** | Asymmetric knee flexion ($|\theta_{left} - \theta_{right}| > 28^\circ$), wide ankle spacing, foot lift $> 0.10$, or sustained net hip displacement across 10 frames. | Normal State |
| **Standing** | Upright torso alignment, extended leg angles ($\ge 150^\circ$), low translational motion. | Normal State |
| **Normal** | Landmark detection confidence $< 0.50$ (no person detected). | System Idle |

---

## ☁️ Deployment on Streamlit Cloud

1. Push your repository to **GitHub** (ensure `app.py`, `pose_landmarker.task`, `logo.png`, and `requirements.txt` are included).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Select your repository, branch (`main`), and set the main file path to `app.py`.
4. Deploy the application.

---

## 🔮 Future Improvements & Maintenance

- **CCTV & RTSP Feed Integration:** Integrate background processing workers for non-browser-based continuous IP camera surveillance.
- **Deep Learning Sequence Modeling:** Pair MediaPipe coordinates with an LSTM or GRU temporal sequence network for transition-state analysis (e.g., distinguishing an intentional floor sit from a rapid collapse).
- **Automated Carer Notification:** Integrate SMS/Email emergency dispatch notifications (via Twilio or SendGrid APIs) when fall incidents trigger.
- **Low-Light / Occlusion Compensation:** Incorporate infrared frame processing or YOLOv8-pose models to maintain reliability during night-time elderly monitoring.