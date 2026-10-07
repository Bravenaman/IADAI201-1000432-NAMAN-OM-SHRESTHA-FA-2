# Fall detection site.
# The activity comes from the body: shoulders, hips, knees, and feet.

import io
import json
import math
import threading
from pathlib import Path

import altair as alt
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
LOGO_PATH = ROOT / "logo.png"

st.set_page_config(page_title="Fall detection", layout="centered")


def current_theme():
    # The address bar remembers the choice after the page reloads.
    saved = st.query_params.get("theme")
    if saved in ("light", "dark"):
        return saved
    detected = st.context.theme.type
    if detected in ("light", "dark"):
        return detected
    return "dark"


def paint_page(theme_name):
    # Same colour bloom in both modes. Light uses a pale base, dark uses the night base.
    if theme_name == "light":
        base = "#fff8f6"
        blooms = """
            radial-gradient(circle at 8% 0%, rgba(255, 59, 48, 0.55), transparent 46%),
            radial-gradient(circle at 96% 6%, rgba(255, 152, 0, 0.50), transparent 44%),
            radial-gradient(circle at 100% 78%, rgba(33, 150, 243, 0.48), transparent 48%),
            radial-gradient(circle at 0% 86%, rgba(156, 39, 176, 0.50), transparent 46%),
            radial-gradient(circle at 48% 42%, rgba(255, 235, 59, 0.42), transparent 40%),
            radial-gradient(circle at 72% 58%, rgba(76, 175, 80, 0.42), transparent 44%)
        """
    else:
        base = "#3a2460"
        blooms = """
            radial-gradient(circle at 18% 12%, rgba(150, 230, 230, 0.55), transparent 26%),
            radial-gradient(circle at 8% 0%, rgba(255, 110, 100, 0.72), transparent 46%),
            radial-gradient(circle at 96% 6%, rgba(255, 176, 70, 0.62), transparent 44%),
            radial-gradient(circle at 100% 78%, rgba(90, 180, 255, 0.58), transparent 48%),
            radial-gradient(circle at 0% 86%, rgba(190, 110, 230, 0.62), transparent 46%),
            radial-gradient(circle at 48% 42%, rgba(255, 235, 120, 0.28), transparent 36%),
            radial-gradient(circle at 72% 58%, rgba(120, 220, 130, 0.42), transparent 42%)
        """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {base};
            background-image: {blooms};
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stMainMenuDivider"]:has(+ [data-testid="stThemeSwitcher"]),
        [data-testid="stThemeSwitcher"] {{
            display: none;
        }}
        /* The logo's built-in expand button only grows the picture. */
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stElementToolbar"] {{
            display: none;
        }}
        /* The about icon sits on the logo and only shows while the pointer is there. */
        [data-testid="stColumn"]:has([data-testid="stImage"]) {{
            position: relative;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stVerticalBlock"] {{
            gap: 0;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stElementContainer"]:has([data-testid="stButton"]) {{
            position: absolute;
            top: 38px;
            left: 40px;
            width: 2.15rem;
            height: 2.15rem;
            margin: 0;
            z-index: 5;
            opacity: 0;
            transition: opacity 0.15s ease;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]):hover [data-testid="stElementContainer"]:has([data-testid="stButton"]) {{
            opacity: 1;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stBaseButton-tertiary"] {{
            width: 2.15rem;
            min-height: 2.15rem;
            padding: 0;
            border-radius: 999px;
            background: #ffffff;
            color: #1a2744;
            border: 1px solid rgba(26, 39, 68, 0.18);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stIconMaterial"] {{
            font-size: 1.2rem;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stBaseButton-tertiary"] p {{
            display: none;
        }}
        [data-testid="stHtml"]:has(#fall-alarm-setup) {{
            display: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def switch_theme(choice):
    # Streamlit reads this saved name on the next page load.
    stored = "Light" if choice == "Light" else "Dark"
    value = stored.lower()
    st.html(
        f"""
        <script>
        const key = "stActiveTheme-" + window.location.pathname + "-v2";
        localStorage.setItem(key, JSON.stringify("{stored}"));
        const url = new URL(window.location.href);
        url.searchParams.set("theme", "{value}");
        window.location.replace(url.toString());
        </script>
        """,
        unsafe_allow_javascript=True,
    )


look = current_theme()
paint_page(look)


@st.dialog("What this AI does")
def show_about():
    st.image(LOGO_PATH, width=120)
    st.write(
        "This site looks at a person's body and names what they are doing. "
        "It uses the shoulders, hips, knees, and feet."
    )
    st.write("You can give it a photo, a video, one camera shot, or the live camera.")
    st.markdown(
        """
- **Fall** — the body has tipped over
- **Off balance** — the body is leaning
- **Sitting**
- **Standing**
- **Walking**
- **Normal** — no person is in the picture
        """
    )
    st.write("A photo or a camera shot gives one answer. A video counts the frames and draws the charts.")


main_col, theme_col = st.columns([4.4, 1.3], vertical_alignment="center")
with main_col:
    logo_col, title_col = st.columns([1, 4.5], vertical_alignment="center")
    with logo_col:
        st.image(LOGO_PATH, width=78)
        if st.button("About", icon=":material/info:", key="logo_about", type="tertiary"):
            show_about()
    with title_col:
        st.title("Fall detection")
with theme_col:
    label = "Light" if look == "light" else "Dark"
    if st.session_state.get("_applied_theme") != look:
        st.session_state.appearance = label
        st.session_state._applied_theme = look
    choice = st.segmented_control(
        "Appearance",
        ["Light", "Dark"],
        key="appearance",
        label_visibility="collapsed",
        required=True,
    )
    if choice and choice.lower() != look:
        st.session_state._applied_theme = choice.lower()
        switch_theme(choice)

st.write("Upload a photo or a video, take one picture, or start the live camera.")
st.session_state["_fall_alarm_shown"] = False


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
        self.fall_hold = False


# The live callback runs on its own thread. MediaPipe has to stay on that thread.
_live_local = threading.local()
_live_label = {"name": None, "false_alarm": None}
_live_matrix = {}
MATRIX_LABELS = ["fall", "sleeping", "sitting", "standing", "walking", "off balance", "normal"]


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


def thigh_span(hip, knee):
    return math.hypot(knee.x - hip.x, knee.y - hip.y)


def knees_toward_camera(landmarks):
    # MediaPipe z is smaller when a point is closer to the camera.
    hip_z = (getattr(landmarks[23], "z", 0.0) + getattr(landmarks[24], "z", 0.0)) / 2
    knee_z = (getattr(landmarks[25], "z", 0.0) + getattr(landmarks[26], "z", 0.0)) / 2
    return hip_z - knee_z > 0.08


def chair_sit(landmarks):
    # On a chair, facing the camera, the thighs point at the lens.
    # A front stand can look short too, but those knees are not closer than the hips.
    def readable(point):
        visibility = point.visibility if point.visibility is not None else 1.0
        return visibility >= 0.35

    if not readable(landmarks[25]) or not readable(landmarks[26]):
        return False
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    torso = hip_y - shoulder_y
    if torso < 0.05:
        return False
    left_drop = landmarks[25].y - hip_y
    right_drop = landmarks[26].y - hip_y
    knees_up = -0.02 < left_drop < torso * 0.60 and -0.02 < right_drop < torso * 0.60
    return knees_up and knees_toward_camera(landmarks)


def looks_like_sleep(landmarks, tilt, left_bend, right_bend):
    # Flat, with both legs straight, and long from shoulder to foot.
    if tilt < 50 or left_bend < 150 or right_bend < 150:
        return False
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    ankle_x = (landmarks[27].x + landmarks[28].x) / 2
    ankle_y = (landmarks[27].y + landmarks[28].y) / 2
    return math.hypot(ankle_x - shoulder_x, ankle_y - shoulder_y) > 0.35


def activity_from_pose(landmarks, last_tilt=None, hips=None):
    # Off balance is a lean, or a lean that is growing. Fall is a bigger lean.
    # A step is one foot ahead, or one knee really bent. Feet apart, from the front, is still a stand.
    tilt, bend, ankle_gap, hip_width, left_bend, right_bend, foot_lift, hip_x, hip_y = body_features(landmarks)
    growing = last_tilt is not None and (tilt - last_tilt) > 12 and tilt > 18
    left_thigh = thigh_span(landmarks[23], landmarks[25])
    right_thigh = thigh_span(landmarks[24], landmarks[26])
    bent_knee = min(left_bend, right_bend) < 150
    # A tiny thigh from the front makes the knee angle wobble, so ignore that noise.
    stepping = (left_thigh > 0.06 or right_thigh > 0.06) and bent_knee and abs(left_bend - right_bend) > 28
    if seen(landmarks[27]) and seen(landmarks[28]):
        # From the side, a step spreads the feet much wider than the hips, even with straight knees.
        # From the front, a normal or wide stance is only a bit wider than the hips.
        stride = ankle_gap > max(0.16, hip_width * 3.0)
        left_reach = abs(landmarks[27].x - landmarks[23].x)
        right_reach = abs(landmarks[28].x - landmarks[24].x)
        reach_gap = abs(left_reach - right_reach)
        stepping = stepping or stride or reach_gap > 0.12 or (foot_lift > 0.08 and reach_gap > 0.05)
        if bent_knee:
            wide_step = ankle_gap > max(0.28, hip_width * 2.6)
            stepping = stepping or wide_step
    moving = hips is not None and hip_is_moving(hips, hip_x, hip_y)
    folded = bend < 130 and max(left_bend, right_bend) < 155 and (left_thigh > 0.05 or right_thigh > 0.05)
    sitting = folded or chair_sit(landmarks)
    # A body that just tipped over is a fall. A body that was already flat is not.
    tipped = last_tilt is not None and last_tilt < 25 and tilt > 35
    false_alarm = None
    if tilt > 35 and not tipped and sitting and tilt < 60:
        # A lean on a chair is the same shape the fall rule looks for.
        # A body flat on the floor is past this, so a real fall still counts.
        false_alarm = "sitting"
        label = "sitting"
        confidence = 0.80
    elif tilt > 35 and not tipped and looks_like_sleep(landmarks, tilt, left_bend, right_bend):
        # Sleeping is flat with straight legs, which is also how a fall looks.
        false_alarm = "sleeping"
        label = "sleeping"
        confidence = 0.80
    elif tilt > 35:
        label = "fall"
        confidence = min(0.99, tilt / 50)
    elif tilt > 20 or growing:
        label = "off balance"
        confidence = min(0.90, max(tilt, 20) / 40)
    elif folded or chair_sit(landmarks):
        label = "sitting"
        confidence = 0.85
    elif stepping or moving:
        label = "walking"
        confidence = 0.85
    else:
        label = "standing"
        confidence = 0.85
    return label, round(float(confidence), 2), tilt, false_alarm


def keep_real_fall(label, false_alarm, tilt, hold):
    # Once the body tips, staying flat is still that fall.
    # Someone who was already lying down never starts this hold.
    if label == "fall" and not false_alarm:
        return "fall", None, True
    if hold and tilt is not None and tilt > 35:
        return "fall", None, True
    return label, false_alarm, False


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


# OpenCV uses blue, green, red order.
ACTIVITY_COLORS = {
    "fall": (0, 0, 255),
    "off balance": (0, 140, 255),
    "sitting": (255, 80, 20),
    "standing": (180, 30, 180),
    "walking": (0, 230, 255),
    "normal": (40, 180, 40),
    "sleeping": (180, 120, 60),
}


def paint_label(frame, label):
    color = ACTIVITY_COLORS.get(label, (40, 180, 40))
    cv2.rectangle(frame, (6, 8), (260, 46), (20, 20, 20), -1)
    cv2.putText(frame, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def predict_frame(pose, frame, last_tilt=None, hips=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = pose.detect(mp_image)
    if not result.pose_landmarks:
        if hips is not None:
            hips.clear()
        paint_label(frame, "normal")
        return frame, "normal", 0.0, None, None
    landmarks = result.pose_landmarks[0]
    draw_pose(frame, landmarks)
    label, confidence, tilt, false_alarm = activity_from_pose(landmarks, last_tilt, hips)
    paint_label(frame, label)
    return frame, label, confidence, tilt, false_alarm


ACTIVITIES = ["fall", "off balance", "sitting", "standing", "walking", "normal"]
BAR_COLORS = {
    "fall": "#ff3b30",
    "off balance": "#ff9800",
    "sitting": "#2196f3",
    "standing": "#9c27b0",
    "walking": "#ffeb3b",
    "normal": "#4caf50",
}


def show_bars(rows, value_name, value_label, top, number_format):
    # The axis stops at the real total, and every activity fits on the page.
    if top < 1:
        top = 1
    labeled = []
    for row in rows:
        number = format(row[value_name], number_format)
        labeled.append(
            {
                "activity": f"{row['activity']} ({number})",
                "kind": row["activity"],
                value_name: row[value_name],
            }
        )
    order = [row["activity"] for row in labeled]
    chart = (
        alt.Chart(alt.Data(values=labeled))
        .mark_bar()
        .encode(
            y=alt.Y(
                "activity:N",
                title="Activity",
                sort=order,
                axis=alt.Axis(labelLimit=220),
            ),
            x=alt.X(
                f"{value_name}:Q",
                title=value_label,
                scale=alt.Scale(domain=[0, top]),
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=ACTIVITIES,
                    range=[BAR_COLORS[name] for name in ACTIVITIES],
                ),
                legend=None,
            ),
        )
        .properties(height=280)
        .configure(background="transparent")
    )
    st.altair_chart(chart, width="stretch")


def show_picture(frame, label):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(rgb, width="stretch")
    color = BAR_COLORS.get(label, "#ffffff")
    st.markdown(
        f"Activity: <span style='color:{color}; font-weight:700'>{label}</span>",
        unsafe_allow_html=True,
    )


def show_confidence(label, confidence):
    show_bars(
        [
            {"activity": name, "confidence": confidence if name == label else 0}
            for name in ACTIVITIES
        ],
        "confidence",
        "Confidence",
        1,
        ".2f",
    )


def prepare_alarm():
    # The browser only allows sound after a click. This remembers that click.
    st.html(
        """
        <div id="fall-alarm-setup"></div>
        <script>
        (function () {
          const first = !window.__fallAlarmReady;
          window.__fallAlarmReady = true;
          if (first) window.__fallAlarmMuted = false;

          function audio() {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return null;
            if (!window.__fallAudio) window.__fallAudio = new Ctx();
            return window.__fallAudio;
          }

          function arm() {
            const ctx = audio();
            if (ctx && ctx.state !== "running") ctx.resume();
          }
          if (first) {
            window.addEventListener("pointerdown", arm, true);
            window.addEventListener("keydown", arm, true);
          }

          function silence() {
            const nodes = window.__fallAlarmNodes;
            if (!nodes) return;
            clearInterval(nodes.timer);
            try { nodes.osc.stop(); } catch (e) {}
            try { nodes.gain.disconnect(); } catch (e) {}
            window.__fallAlarmNodes = null;
          }

          window.__playFallAlarm = function () {
            if (window.__fallAlarmMuted) return false;
            const ctx = audio();
            if (!ctx) return false;
            if (ctx.state !== "running") return false;
            if (window.__fallAlarmNodes) return true;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "square";
            osc.frequency.value = 880;
            gain.gain.value = 0.12;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            let high = false;
            const timer = setInterval(function () {
              high = !high;
              osc.frequency.setValueAtTime(high ? 880 : 520, ctx.currentTime);
            }, 320);
            window.__fallAlarmNodes = { osc: osc, gain: gain, timer: timer };
            return true;
          };

          window.__stopFallAlarm = function (allowNext) {
            silence();
            window.__fallAlarmMuted = !allowNext;
          };

          window.__armAlarm = function (token) {
            if (token && window.__alarmToken !== token) {
              window.__alarmToken = token;
              window.__fallAlarmMuted = false;
            }
            if (window.__playFallAlarm) return window.__playFallAlarm();
            return false;
          };
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def quiet_alarm():
    st.html(
        """
        <div id="fall-alarm-setup"></div>
        <script>
        if (window.__stopFallAlarm) window.__stopFallAlarm(true);
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def sound_alarm(token):
    # Snooze stops the tone immediately. A new photo, video, or camera fall can ring again.
    # The live camera keeps the same token, so snooze stays on until that fall is over.
    safe_token = json.dumps(token)
    st.html(
        f"""
        <button type="button" id="fall-snooze" style="display:block;width:100%;margin:0.2rem 0 0.8rem 0;background:#ffd60a;color:#1a1a1a;border:0;border-radius:0.6rem;padding:0.85rem 1rem;font:800 1.15rem Nunito,sans-serif;cursor:pointer;">
          Snooze alarm
        </button>
        <p id="fall-alarm-note">An alarm is sounding. Press Snooze alarm to shut it off.</p>
        <script>
        (function () {{
          const note = document.getElementById("fall-alarm-note");
          const btn = document.getElementById("fall-snooze");
          if (window.__armAlarm) window.__armAlarm({safe_token});
          function refresh() {{
            const on = !!window.__fallAlarmNodes;
            const quiet = !!window.__fallAlarmMuted;
            if (!btn) return;
            if (on) {{
              btn.textContent = "Snooze alarm";
              btn.style.background = "#ffd60a";
              btn.style.color = "#1a1a1a";
              if (note) note.textContent = "An alarm is sounding. Press Snooze alarm to shut it off.";
            }} else if (quiet) {{
              btn.textContent = "Alarm snoozed";
              btn.style.background = "#d7d7d7";
              btn.style.color = "#1a1a1a";
              if (note) note.textContent = "Snoozed. The sound is off.";
            }} else {{
              btn.textContent = "Play alarm";
              btn.style.background = "#ffd60a";
              btn.style.color = "#1a1a1a";
              if (note) note.textContent = "Press Play alarm if the sound did not start.";
            }}
          }}
          if (btn) {{
            btn.onclick = function () {{
              if (window.__fallAlarmNodes) {{
                window.__stopFallAlarm(false);
                refresh();
                return;
              }}
              if (window.__fallAlarmMuted) return;
              window.__fallAlarmMuted = false;
              const Ctx = window.AudioContext || window.webkitAudioContext;
              if (!window.__fallAudio && Ctx) window.__fallAudio = new Ctx();
              const ctx = window.__fallAudio;
              const go = function () {{
                if (window.__playFallAlarm) window.__playFallAlarm();
                refresh();
              }};
              if (ctx && ctx.state !== "running") ctx.resume().then(go);
              else go();
            }};
          }}
          refresh();
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def show_emergency(token="fall"):
    # One alarm panel per run, even if a video has many fall frames.
    if st.session_state.get("_fall_alarm_shown"):
        return
    st.session_state["_fall_alarm_shown"] = True
    st.markdown(
        """
        <style>
        @keyframes fallAlarm {
            0%, 100% { opacity: 0.55; }
            50% { opacity: 1; }
        }
        .fall-alarm {
            pointer-events: none;
            position: fixed;
            inset: 0;
            z-index: 999990;
            box-shadow:
                inset 0 0 0 18px rgba(255, 0, 0, 0.95),
                inset 0 0 110px rgba(255, 0, 0, 0.72);
            animation: fallAlarm 0.7s ease-in-out infinite;
        }
        </style>
        <div class="fall-alarm"></div>
        """,
        unsafe_allow_html=True,
    )
    st.error("EMERGENCY: A fall was detected.")
    sound_alarm(token)
    st.subheader("Do this now")
    st.markdown(
        """
1. Stay with the person. Do not leave them alone.
2. Call someone close by, then call **112**.
3. Do not pull them up if they might be hurt.
4. Say that someone has fallen, and give the address.
5. Keep them warm and keep talking until help arrives.
        """
    )
    st.write("Emergency number: **112**")
    call_col, hospital_col = st.columns(2)
    with call_col:
        st.link_button("Call 112", "tel:112")
    with hospital_col:
        st.link_button(
            "Nearest hospital",
            "https://www.google.com/maps/search/?api=1&query=nearest+hospital",
        )
    st.write("Press Call 112 so the phone rings the emergency number. This page cannot dial it by itself.")
    st.write("Open the nearest hospital and tell them someone has fallen and needs help.")


def add_count(counts, label, false_alarm):
    # A false alarm was called a fall, then checked as sitting or sleeping.
    checked = false_alarm or label or "normal"
    called = "fall" if false_alarm else (label or "normal")
    if checked not in MATRIX_LABELS:
        checked = "normal"
    if called not in MATRIX_LABELS:
        called = "normal"
    counts[(checked, called)] = counts.get((checked, called), 0) + 1


def show_heatmap(counts):
    # Same idea as the Colab heatmap. This grid is only the file or camera that is open now.
    st.subheader("Confusion matrix")
    st.write("Rows are what the check decided. Columns are what the fall rule called. Darker means more frames.")
    rows = []
    top = 1
    for checked in MATRIX_LABELS:
        for called in MATRIX_LABELS:
            count = int(counts.get((checked, called), 0))
            top = max(top, count)
            rows.append({"Checked": checked, "Called": called, "Frames": count})
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("Called:N", sort=MATRIX_LABELS, title="Called"),
        y=alt.Y("Checked:N", sort=MATRIX_LABELS, title="Checked"),
    )
    heat = base.mark_rect().encode(
        color=alt.Color(
            "Frames:Q",
            scale=alt.Scale(domain=[0, top], scheme="reds"),
            legend=alt.Legend(title="Frames"),
        ),
        tooltip=[
            alt.Tooltip("Checked:N", title="Checked"),
            alt.Tooltip("Called:N", title="Called"),
            alt.Tooltip("Frames:Q", title="Frames"),
        ],
    )
    numbers = base.mark_text().encode(
        text=alt.Text("Frames:Q"),
        color=alt.condition(
            alt.datum.Frames > top / 2,
            alt.value("white"),
            alt.value("#31333f"),
        ),
    )
    chart = (heat + numbers).properties(height=340).configure(background="transparent")
    st.altair_chart(chart, width="stretch")


def show_false_alarm(kind, sitting_count=None, sleeping_count=None):
    if sitting_count is not None:
        st.warning(
            f"False alarms caught: {sitting_count} sitting, {sleeping_count} sleeping. "
            "Those frames looked like a fall, so the emergency stayed off for them."
        )
        return
    if kind == "sleeping":
        st.warning(
            "False alarm: this looks like sleeping, not a fall. "
            "A person lying down is flat, the same shape as a fall, so the emergency stays off."
        )
    else:
        st.warning(
            "False alarm: this looks like sitting, not a fall. "
            "The body is leaning, but the knees are bent like a chair, so the emergency stays off."
        )


def show_result_message(label, false_alarm=None, token="fall"):
    counts = {}
    add_count(counts, label, false_alarm)
    show_heatmap(counts)
    if false_alarm:
        show_false_alarm(false_alarm)
    elif label == "fall":
        show_emergency(token)
    elif label == "off balance":
        st.warning("Warning: the person is losing balance.")
    elif label == "normal":
        st.info("No person found in this frame.")
    else:
        st.success("No fall detected.")


def show_result(frame, label, confidence, false_alarm=None, token="fall"):
    show_picture(frame, label)
    if label in ACTIVITIES:
        show_confidence(label, confidence)
    show_result_message(label, false_alarm, token)


def frames_from_video(data):
    # Read a few frames from the upload. A long video stays quick.
    # PyAV reads the file in memory, so the app folder does not need to be writable.
    frames = []
    try:
        container = av.open(io.BytesIO(data))
    except Exception:
        return frames
    try:
        stream = container.streams.video[0]
        total = stream.frames or 0
        if total < 1 and stream.duration and stream.average_rate:
            total = int(stream.duration * stream.time_base * stream.average_rate)
        step = max(1, total // 20) if total > 0 else 1
        index = 0
        for frame in container.decode(stream):
            if index % step == 0:
                frames.append(frame.to_ndarray(format="bgr24"))
            index += 1
            if len(frames) >= 20:
                break
    except Exception:
        return frames
    finally:
        container.close()
    return frames


pose = load_pose()
prepare_alarm()
mode = st.radio("Choose an input", ["Photo", "Video", "Camera", "Live"], horizontal=True)

if mode == "Photo":
    photo = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
    if photo is not None:
        frame = read_image_bytes(photo.getvalue())
        if frame is None:
            st.error("That file could not be read as a photo.")
        else:
            frame, label, confidence, _tilt, false_alarm = predict_frame(pose, frame)
            show_result(frame, label, confidence, false_alarm, f"photo-{photo.name}-{photo.size}")

elif mode == "Video":
    video = st.file_uploader("Video", type=["mp4", "avi", "mov"])
    if video is not None:
        try:
            frames = frames_from_video(video.getvalue())
        except Exception:
            frames = []
        if len(frames) == 0:
            st.error("That video could not be read.")
        else:
            counts = {name: 0 for name in ACTIVITIES}
            false_counts = {"sitting": 0, "sleeping": 0}
            matrix = {}
            last = None
            hold = False
            last_tilt = None
            try:
                for frame in frames:
                    frame, label, confidence, tilt, false_alarm = predict_frame(
                        pose, frame.copy(), last_tilt
                    )
                    label, false_alarm, hold = keep_real_fall(label, false_alarm, tilt, hold)
                    paint_label(frame, label)
                    last_tilt = tilt
                    if false_alarm:
                        false_counts[false_alarm] += 1
                    if label in counts:
                        counts[label] += 1
                    add_count(matrix, label, false_alarm)
                    last = (frame, label, confidence, false_alarm)
            except Exception:
                st.error("That video could not be read.")
            else:
                frame, label, confidence, false_alarm = last
                if label in ACTIVITIES:
                    show_confidence(label, confidence)
                show_picture(frame, label)
                st.write("Frames checked:", len(frames))
                show_bars(
                    [{"activity": name, "frames": counts[name]} for name in ACTIVITIES],
                    "frames",
                    "Frames",
                    len(frames),
                    ".0f",
                )
                caught = false_counts["sitting"] + false_counts["sleeping"]
                show_heatmap(matrix)
                if caught:
                    show_false_alarm(
                        None,
                        sitting_count=false_counts["sitting"],
                        sleeping_count=false_counts["sleeping"],
                    )
                if counts["fall"] > 0:
                    show_emergency(f"video-{video.name}-{video.size}")
                elif not false_alarm:
                    if label == "off balance":
                        st.warning("Warning: the person is losing balance.")
                    elif label == "normal":
                        st.info("No person found in this frame.")
                    else:
                        st.success("No fall detected.")

elif mode == "Camera":
    camera = st.camera_input("Take a picture")
    if camera is not None:
        frame = read_image_bytes(camera.getvalue())
        if frame is None:
            st.error("The camera picture could not be read.")
        else:
            frame, label, confidence, _tilt, false_alarm = predict_frame(pose, frame)
            show_result(frame, label, confidence, false_alarm, f"camera-{len(camera.getvalue())}")

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
            image, label, _confidence, tilt, false_alarm = predict_frame(
                state.pose, image, state.last_tilt, state.hips
            )
            label, false_alarm, state.fall_hold = keep_real_fall(
                label, false_alarm, tilt, state.fall_hold
            )
            state.last_tilt = tilt
            paint_label(image, label)
            _live_label["name"] = label
            _live_label["false_alarm"] = false_alarm
            add_count(_live_matrix, label, false_alarm)
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        except Exception as error:
            print("Live frame skipped:", error)
            return frame

    live_camera = webrtc_streamer(
        key="live-camera",
        video_frame_callback=on_frame,
        media_stream_constraints={
            "video": {"width": {"ideal": 480}, "height": {"ideal": 360}, "facingMode": "user"},
            "audio": False,
        },
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_html_attrs={"autoPlay": True, "muted": True, "playsInline": True, "controls": False},
        media_toggle_controls=False,
        sendback_audio=False,
        async_processing=True,
    )

    @st.fragment(run_every=1)
    def live_fall_alarm():
        # Only the current live frame can open this. Stop clears a previous fall.
        if not live_camera.state.playing:
            _live_label["name"] = None
            _live_label["false_alarm"] = None
            _live_matrix.clear()
            quiet_alarm()
            return
        st.session_state["_fall_alarm_shown"] = False
        kind = _live_label.get("false_alarm")
        name = _live_label.get("name")
        if _live_matrix:
            show_heatmap(dict(_live_matrix))
        if name == "fall":
            show_emergency("live")
        elif kind:
            show_false_alarm(kind)
            quiet_alarm()
        else:
            quiet_alarm()

    live_fall_alarm()

if not st.session_state.get("_fall_alarm_shown"):
    quiet_alarm()
