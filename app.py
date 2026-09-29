# Fall detection site.
# The activity comes from the body: shoulders, hips, knees, and feet.

import math
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer

ROOT = Path(__file__).parent
TASK_PATH = ROOT / "pose_landmarker.task"

st.set_page_config(page_title="Fall detection", layout="centered")
st.title("Fall detection")
st.write("Upload a photo or a video, take one picture, or start the live camera.")


def make_pose():
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(TASK_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


@st.cache_resource
def load_pose():
    return make_pose()


class LiveCamera:
    def __init__(self):
        self.pose = make_pose()
        self.last_tilt = None
        self.hips = []


# The live callback runs on its own thread. MediaPipe has to stay on that thread.
_live_local = threading.local()


def live_state():
    state = getattr(_live_local, "state", None)
    if state is None:
        state = LiveCamera()
        _live_local.state = state
    return state


def knee_bend(hip, knee, ankle):
    ax, ay = hip.x - knee.x, hip.y - knee.y
    bx, by = ankle.x - knee.x, ankle.y - knee.y
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na * nb == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cos))


def seen(point):
    # A missing foot should not count as a step.
    visibility = point.visibility if point.visibility is not None else 1.0
    return visibility >= 0.5


def body_features(landmarks):
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    left_bend = knee_bend(landmarks[23], landmarks[25], landmarks[27])
    right_bend = knee_bend(landmarks[24], landmarks[26], landmarks[28])
    bend = (left_bend + right_bend) / 2
    ankle_gap = abs(landmarks[27].x - landmarks[28].x)
    hip_width = abs(landmarks[23].x - landmarks[24].x)
    foot_lift = abs(landmarks[27].y - landmarks[28].y)
    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    tilt = abs(math.degrees(math.atan2(dx, dy))) if (dx or dy) else 0.0
    return tilt, bend, ankle_gap, hip_width, left_bend, right_bend, foot_lift, hip_x, hip_y


def hip_is_moving(hips, hip_x, hip_y):
    # A walk toward the camera keeps the feet close, but the hips still travel.
    hips.append((hip_x, hip_y))
    if len(hips) > 10:
        del hips[0]
    if len(hips) < 6:
        return False
    path = 0.0
    for start, end in zip(hips, hips[1:]):
        path += math.hypot(end[0] - start[0], end[1] - start[1])
    net = math.hypot(hips[-1][0] - hips[0][0], hips[-1][1] - hips[0][1])
    # Shifting weight stays in place. A walk keeps moving the same way.
    return path > 0.10 and net > path * 0.45


def activity_from_pose(landmarks, last_tilt=None, hips=None):
    # Off balance is a lean, or a lean that is growing. Fall is a bigger lean.
    # Walking is a step: feet apart, one foot up, one knee bent, or the hips moving.
    tilt, bend, ankle_gap, hip_width, left_bend, right_bend, foot_lift, hip_x, hip_y = body_features(landmarks)
    growing = last_tilt is not None and (tilt - last_tilt) > 12 and tilt > 18
    # A normal stance is about as wide as the hips. A step is wider, or one foot is up.
    stepping = abs(left_bend - right_bend) > 28
    if seen(landmarks[27]) and seen(landmarks[28]):
        stepping = stepping or ankle_gap > max(0.22, hip_width * 2.2) or foot_lift > 0.06
    moving = hips is not None and hip_is_moving(hips, hip_x, hip_y)
    if tilt > 35:
        label = "fall"
        confidence = min(0.99, tilt / 50)
    elif tilt > 20 or growing:
        label = "off balance"
        confidence = min(0.90, max(tilt, 20) / 40)
    elif bend < 130 and max(left_bend, right_bend) < 155:
        label = "sitting"
        confidence = 0.85
    elif stepping or moving:
        label = "walking"
        confidence = 0.85
    else:
        label = "standing"
        confidence = 0.85
    return label, round(float(confidence), 2), tilt


def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]
    points = []
    links = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    for a, b in links:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)


def read_image_bytes(data):
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame


def paint_label(frame, label):
    if label == "fall":
        color = (0, 0, 255)
    elif label == "off balance":
        color = (0, 140, 255)
    else:
        color = (0, 180, 0)
    cv2.putText(frame, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def predict_frame(pose, frame, last_tilt=None, hips=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = pose.detect(mp_image)
    if not result.pose_landmarks:
        if hips is not None:
            hips.clear()
        paint_label(frame, "normal")
        return frame, "normal", 0.0, None
    landmarks = result.pose_landmarks[0]
    draw_pose(frame, landmarks)
    label, confidence, tilt = activity_from_pose(landmarks, last_tilt, hips)
    paint_label(frame, label)
    return frame, label, confidence, tilt


def show_result(frame, label, confidence):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(rgb, use_container_width=True)
    st.write("Activity:", label)
    st.write("Confidence:", confidence)
    if label == "fall":
        st.error("EMERGENCY: Fall detected. Please check on the person.")
    elif label == "off balance":
        st.warning("Warning: the person is losing balance.")
    elif label == "normal":
        st.info("No person found in this frame.")
    else:
        st.success("No fall detected.")


def frames_from_video(data):
    # Save the upload, then read a few frames so a long video stays quick.
    temp_path = ROOT / "_upload_video.mp4"
    temp_path.write_bytes(data)
    cap = cv2.VideoCapture(str(temp_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        total = 1
    step = max(1, total // 20)
    frames = []
    for index in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        if len(frames) >= 20:
            break
    cap.release()
    temp_path.unlink(missing_ok=True)
    return frames


pose = load_pose()
mode = st.radio("Choose an input", ["Photo", "Video", "Camera", "Live"], horizontal=True)

if mode == "Photo":
    photo = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
    if photo is not None:
        frame = read_image_bytes(photo.getvalue())
        if frame is None:
            st.error("That file could not be read as a photo.")
        else:
            frame, label, confidence, _tilt = predict_frame(pose, frame)
            show_result(frame, label, confidence)

elif mode == "Video":
    video = st.file_uploader("Video", type=["mp4", "avi", "mov"])
    if video is not None:
        frames = frames_from_video(video.getvalue())
        if len(frames) == 0:
            st.error("That video could not be read.")
        else:
            counts = {"fall": 0, "off balance": 0, "sitting": 0, "standing": 0, "walking": 0, "normal": 0}
            last = None
            for frame in frames:
                frame, label, confidence, _tilt = predict_frame(pose, frame.copy())
                counts[label] += 1
                last = (frame, label, confidence)
            st.write("Frames checked:", len(frames))
            st.write("Counts:", counts)
            show_result(*last)
            if counts["fall"] > 0:
                st.error("EMERGENCY: A fall showed up in this video.")

elif mode == "Camera":
    camera = st.camera_input("Take a picture")
    if camera is not None:
        frame = read_image_bytes(camera.getvalue())
        if frame is None:
            st.error("The camera picture could not be read.")
        else:
            frame, label, confidence, _tilt = predict_frame(pose, frame)
            show_result(frame, label, confidence)

else:
    st.write("Press Start and allow the camera. Click Stop when you are done.")
    st.write("If the picture stays black, press Stop, refresh this page, then press Start again.")

    def on_frame(frame):
        # Always send a picture back. A pose error must not turn the camera black.
        try:
            image = np.ascontiguousarray(frame.to_ndarray(format="bgr24"))
            if image.size == 0:
                return frame
            state = live_state()
            image, _label, _confidence, state.last_tilt = predict_frame(
                state.pose, image, state.last_tilt, state.hips
            )
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        except Exception as error:
            print("Live frame skipped:", error)
            return frame

    webrtc_streamer(
        key="live-camera",
        video_frame_callback=on_frame,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "facingMode": "user"},
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_html_attrs={"autoPlay": True, "muted": True, "playsInline": True, "controls": False},
        media_toggle_controls=False,
        sendback_audio=False,
        async_processing=False,
    )
