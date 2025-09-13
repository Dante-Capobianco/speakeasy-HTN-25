from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import argparse
import json
import os
import tempfile
import requests
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from anyio import to_thread

app = FastAPI()

# Hand Landmarker model (download on first use)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_LOCAL = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Face Landmarker model (with blendshapes)
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_LOCAL = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Pose Landmarker model (full)
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
POSE_MODEL_LOCAL = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

# Gesture Recognizer model
GESTURE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
GESTURE_MODEL_LOCAL = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")


# -------- Request schema --------
class NonverbalRequest(BaseModel):
    videoUrl: str                      # https(s) URL or local path
    sampleEveryNFrames: int = 10       # sample frequency
    maxFrames: Optional[int] = None    # cap sampled frames
    modules: Optional[List[str]] = None  # e.g. ["hands"]; None => ["hands"]


# -------- Utilities --------
def ensure_model(path: str = MODEL_LOCAL) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(MODEL_URL, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def ensure_model_from(url: str, path: str) -> str:
    """Download a model once and cache it locally."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def to_local_file(path_or_url: str) -> str:
    """If http(s), download to a temp file and return local path; else return input."""
    if path_or_url.lower().startswith(("http://", "https://")):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with requests.get(path_or_url, stream=True, timeout=600) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
        tmp.close()
        return tmp.name
    return path_or_url


# -------- Feedback helpers (English short descriptors) --------
DEFAULT_LEVEL_LABELS = [
    "none",
    "little",
    "medium",
    "relatively many",
    "very much",
]

def label_by_edges(x: float, edges: List[float], labels: List[str] = DEFAULT_LEVEL_LABELS) -> str:
    """Map a numeric value to qualitative labels using ascending edges.
    Bins: x <= 0 -> labels[0]; 0 < x < edges[0] -> labels[1];
          edges[0] <= x < edges[1] -> labels[2];
          edges[1] <= x < edges[2] -> labels[3];
          x >= edges[2] -> labels[4].
    """
    try:
        x = float(x)
    except Exception:
        return labels[0]
    if x <= 0:
        return labels[0]
    if len(edges) < 3:
        # pad if needed
        edges = list(edges) + [edges[-1]] * (3 - len(edges))
    if x < edges[0]:
        return labels[1]
    if x < edges[1]:
        return labels[2]
    if x < edges[2]:
        return labels[3]
    return labels[4]

def label_count(x: float) -> str:
    # Example bins: (0,1), [1,4), [4,8), >=8
    return label_by_edges(x, [1.0, 4.0, 8.0])

def label_ratio(x: float) -> str:
    # Ratio bins tuned for activity share
    return label_by_edges(x, [0.05, 0.20, 0.50])

def label_intensity(x: float) -> str:
    # Movement intensity (torso lengths per second) heuristic bins
    return label_by_edges(x, [0.05, 0.15, 0.30])


# -------- Analyzer interface --------
class Analyzer:
    name: str = "base"
    def start(self, video_props: Dict[str, Any]) -> None:
        pass
    def process(self, frame_rgb, frame_index: int, timestamp_ms: int) -> Optional[Dict[str, Any]]:
        return None
    def finalize(self) -> Dict[str, Any]:
        return {}


# -------- Hands analyzer (MediaPipe Tasks) --------
class HandsAnalyzer(Analyzer):
    name: str = "hands"

    def __init__(self) -> None:
        self.detector = None

    def start(self, video_props: Dict[str, Any]) -> None:
        model_path = ensure_model(MODEL_LOCAL)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def process(self, frame_rgb, frame_index: int, timestamp_ms: int) -> Optional[Dict[str, Any]]:
        # Wrap numpy RGB frame to mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        det = self.detector.detect_for_video(mp_image, timestamp_ms)

        hands_out = []
        if det.hand_landmarks:
            for i, lms in enumerate(det.hand_landmarks):
                pts = [{"x": float(p.x), "y": float(p.y), "z": float(p.z)} for p in lms]
                handed, score = None, None
                if det.handedness and len(det.handedness) > i and len(det.handedness[i]) > 0:
                    handed = det.handedness[i][0].category_name
                    score = float(det.handedness[i][0].score)
                hands_out.append({
                    "handedness": handed,
                    "handedness_score": score,
                    "landmarks": pts,
                })

        return {"hands": hands_out}

    def finalize(self) -> Dict[str, Any]:
        # Placeholder for future aggregate metrics across frames
        return {}


# -------- Face analyzer (MediaPipe Tasks) --------
class FaceAnalyzer(Analyzer):
    name: str = "face"

    def __init__(self) -> None:
        self.detector = None
        # Time-weighted aggregation
        self.dt_s = 0.0
        self.total_time_s = 0.0
        self.metrics = {
            "smile": {"sum": 0.0, "peak": 0.0, "active": 0.0},
            "jawOpen": {"sum": 0.0, "peak": 0.0, "active": 0.0},
            "eyeBlinkLeft": {"sum": 0.0, "peak": 0.0, "active": 0.0},
            "eyeBlinkRight": {"sum": 0.0, "peak": 0.0, "active": 0.0},
        }
        # Per-metric activity thresholds tuned to typical ranges
        # You can adjust these if needed
        self.thresholds = {
            "smile": 0.01,         # smiles can be subtle on blendshapes
            "jawOpen": 0.20,       # talking often ~0.2-0.4
            "eyeBlinkLeft": 0.20,  # blinks are brief; keep threshold modest
            "eyeBlinkRight": 0.20,
        }
        # Blink event detection state (combined both eyes)
        self.blink_count = 0
        self._blink_active = False

    def start(self, video_props: Dict[str, Any]) -> None:
        # Compute per-sample time delta (seconds)
        ms_per_frame = float(video_props.get("ms_per_frame", 33.3))
        sample_n = int(video_props.get("sample_n", 1))
        self.dt_s = (ms_per_frame * max(1, sample_n)) / 1000.0

        model_path = ensure_model_from(FACE_MODEL_URL, FACE_MODEL_LOCAL)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def _accumulate(self, name: str, score: float) -> None:
        m = self.metrics[name]
        m["sum"] += float(score) * self.dt_s
        m["peak"] = max(m["peak"], float(score))
        thr = float(self.thresholds.get(name, 0.2))
        if score >= thr:
            m["active"] += self.dt_s

    def process(self, frame_rgb, frame_index: int, timestamp_ms: int) -> Optional[Dict[str, Any]]:
        # Convert to mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self.detector.detect_for_video(mp_image, timestamp_ms)

        # Default scores when no face detected in this sample
        smile = 0.0
        jaw_open = 0.0
        blink_l = 0.0
        blink_r = 0.0

        if res and getattr(res, "face_blendshapes", None) and len(res.face_blendshapes) > 0:
            # Build category->score map for the first face
            cat_scores: Dict[str, float] = {}
            for cat in res.face_blendshapes[0]:
                cat_scores[cat.category_name] = float(cat.score)

            # Smile: average of mouthSmileLeft/Right if available
            smile_keys = [
                cat_scores.get("mouthSmileLeft"),
                cat_scores.get("mouthSmileRight"),
            ]
            smile_vals = [v for v in smile_keys if v is not None]
            if len(smile_vals) == 2:
                smile = (smile_vals[0] + smile_vals[1]) / 2.0
            elif len(smile_vals) == 1:
                smile = smile_vals[0]

            jaw_open = cat_scores.get("jawOpen", 0.0)
            blink_l = cat_scores.get("eyeBlinkLeft", 0.0)
            blink_r = cat_scores.get("eyeBlinkRight", 0.0)

        # Blink event detection using combined blink intensity (either eye)
        combined_blink = max(blink_l, blink_r)
        blink_thr = float(self.thresholds.get("eyeBlinkLeft", 0.20))
        if combined_blink >= blink_thr:
            if not self._blink_active:
                self._blink_active = True
                self.blink_count += 1
        else:
            self._blink_active = False

        # Accumulate for all sampled frames (treat no-detection as 0)
        self._accumulate("smile", smile)
        self._accumulate("jawOpen", jaw_open)
        self._accumulate("eyeBlinkLeft", blink_l)
        self._accumulate("eyeBlinkRight", blink_r)
        self.total_time_s += self.dt_s

        # No per-frame payload needed for 'face'
        return None

    def finalize(self) -> Dict[str, Any]:
        total = max(self.total_time_s, 1e-9)
        def pack(name: str) -> Dict[str, float]:
            m = self.metrics[name]
            return {
                "avg": m["sum"] / total,
                "peak": m["peak"],
                "active_duration_s": m["active"],
            }

        smile = pack("smile")
        jaw = pack("jawOpen")
        ebl = pack("eyeBlinkLeft")
        ebr = pack("eyeBlinkRight")
        blink_count = int(self.blink_count)

        # Short English feedback
        feedback = {
            "smile_feedback": f"Smiling intensity: {label_ratio(smile['avg'])}",
            "speech_mouth_open_feedback": f"Speaking activity: {label_ratio(jaw['active_duration_s'] / max(self.total_time_s, 1e-9))}",
            "blinking_feedback": f"Blinking frequency: {label_count(blink_count)}",
        }

        return {
            "smile": smile,
            "jawOpen": jaw,
            "eyeBlinkLeft": ebl,
            "eyeBlinkRight": ebr,
            "blinkCount": blink_count,
            "feedback": feedback,
        }


# -------- Pose analyzer (MediaPipe Tasks) --------
class PoseAnalyzer(Analyzer):
    name: str = "pose"

    def __init__(self) -> None:
        self.detector = None
        # Stats accumulation
        self.dt_s = 0.0
        self.total_time_s = 0.0
        self.pose_time_s = 0.0  # time with valid pose
        # Posture tilt (degrees)
        self.tilt_sum = 0.0
        self.tilt_peak = 0.0
        # Stability (center sway std) accumulators
        self.center_x_sum = 0.0
        self.center_y_sum = 0.0
        self.center_x2_sum = 0.0
        self.center_y2_sum = 0.0
        self.center_n = 0
        # Motion intensity
        self.prev_center = None  # (x,y)
        self.prev_torso_len = None
        self.motion_sum = 0.0
        self.motion_peak = 0.0
        self.still_time_s = 0.0
        self.motion_threshold = 0.5  # torso lengths per second
        # Space usage
        self.area_sum = 0.0
        self.area_peak = 0.0
        self.out_of_frame_time_s = 0.0

    def start(self, video_props: Dict[str, Any]) -> None:
        model_path = ensure_model_from(POSE_MODEL_URL, POSE_MODEL_LOCAL)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.PoseLandmarker.create_from_options(options)
        # dt per sample
        ms_per_frame = float(video_props.get("ms_per_frame", 33.3))
        sample_n = int(video_props.get("sample_n", 1))
        self.dt_s = (ms_per_frame * max(1, sample_n)) / 1000.0

    def process(self, frame_rgb, frame_index: int, timestamp_ms: int) -> Optional[Dict[str, Any]]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self.detector.detect_for_video(mp_image, timestamp_ms)

        out = {"landmarks": []}
        if res and getattr(res, "pose_landmarks", None):
            # Use the first detected pose
            if len(res.pose_landmarks) > 0:
                pts = [
                    {"x": float(p.x), "y": float(p.y), "z": float(p.z)}
                    for p in res.pose_landmarks[0]
                ]
                out["landmarks"] = pts

                # ----- Compute pose-based aggregates -----
                lm = res.pose_landmarks[0]
                # landmark indices (MediaPipe Pose):
                # 11 LShoulder, 12 RShoulder, 23 LHip, 24 RHip, 15 LWrist, 16 RWrist, 0 Nose
                try:
                    ls = lm[11]; rs = lm[12]; lh = lm[23]; rh = lm[24]
                except Exception:
                    ls = rs = lh = rh = None

                if ls and rs and lh and rh:
                    mid_shoulders = ((ls.x + rs.x)/2.0, (ls.y + rs.y)/2.0)
                    mid_hips = ((lh.x + rh.x)/2.0, (lh.y + rh.y)/2.0)
                    # Tilt: shoulder line vs horizontal
                    dx = (rs.x - ls.x); dy = (rs.y - ls.y)
                    import math
                    tilt_deg = abs(math.degrees(math.atan2(dy, dx)))
                    self.tilt_sum += tilt_deg * self.dt_s
                    self.tilt_peak = max(self.tilt_peak, tilt_deg)

                    # Center for stability
                    cx = (mid_shoulders[0] + mid_hips[0]) / 2.0
                    cy = (mid_shoulders[1] + mid_hips[1]) / 2.0
                    self.center_x_sum += cx
                    self.center_y_sum += cy
                    self.center_x2_sum += cx * cx
                    self.center_y2_sum += cy * cy
                    self.center_n += 1

                    # Motion intensity: center speed normalized by torso length
                    torso_len = math.hypot(mid_shoulders[0] - mid_hips[0], mid_shoulders[1] - mid_hips[1])
                    if self.prev_center is not None and self.prev_torso_len is not None and self.dt_s > 0 and torso_len > 1e-6:
                        dx_c = cx - self.prev_center[0]
                        dy_c = cy - self.prev_center[1]
                        dist = math.hypot(dx_c, dy_c)
                        # per-second speed in torso lengths
                        speed = (dist / max(torso_len, 1e-6)) / self.dt_s
                        self.motion_sum += speed * self.dt_s
                        self.motion_peak = max(self.motion_peak, speed)
                        if speed < self.motion_threshold:
                            self.still_time_s += self.dt_s
                    self.prev_center = (cx, cy)
                    self.prev_torso_len = torso_len
                    self.pose_time_s += self.dt_s

                # Space usage: bbox area of all pose landmarks
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                xmin, xmax = max(0.0, min(xs)), min(1.0, max(xs))
                ymin, ymax = max(0.0, min(ys)), min(1.0, max(ys))
                area = max(0.0, (xmax - xmin)) * max(0.0, (ymax - ymin))
                self.area_sum += area * self.dt_s
                self.area_peak = max(self.area_peak, area)
                # Out of frame if any landmark is outside [0,1]
                if (min(xs) < 0.0 or max(xs) > 1.0 or min(ys) < 0.0 or max(ys) > 1.0):
                    self.out_of_frame_time_s += self.dt_s

        # always track total time (sampled)
        self.total_time_s += self.dt_s

        return {"pose": out}

    def finalize(self) -> Dict[str, Any]:
        total = max(self.total_time_s, 1e-9)
        pose_time = max(self.pose_time_s, 1e-9)
        # center std
        if self.center_n > 1:
            mean_x = self.center_x_sum / self.center_n
            mean_y = self.center_y_sum / self.center_n
            var_x = max(0.0, self.center_x2_sum / self.center_n - mean_x * mean_x)
            var_y = max(0.0, self.center_y2_sum / self.center_n - mean_y * mean_y)
            import math
            std_x = math.sqrt(var_x)
            std_y = math.sqrt(var_y)
        else:
            std_x = std_y = 0.0

        # Averages are time-weighted by pose_time_s
        posture = {
            "tilt_deg_avg": self.tilt_sum / pose_time,
            "tilt_deg_peak": self.tilt_peak,
        }
        stability = {
            "center_sway_std_x": std_x,
            "center_sway_std_y": std_y,
            "stillness_ratio": min(1.0, self.still_time_s / max(self.pose_time_s, 1e-9)),
        }
        space_use = {
            "bbox_area_avg": self.area_sum / pose_time,
            "bbox_area_peak": self.area_peak,
            "out_of_frame_ratio": min(1.0, self.out_of_frame_time_s / pose_time),
        }
        motion = {
            "movement_intensity_avg": self.motion_sum / pose_time,
            "movement_intensity_peak": self.motion_peak,
            "active_duration_s": self.pose_time_s - self.still_time_s,
        }
        # Short English feedback
        posture_level = "upright posture" if posture["tilt_deg_avg"] >= 170 else ("slight tilt" if posture["tilt_deg_avg"] >= 155 else "noticeable tilt")
        stability_level = "very stable" if stability["stillness_ratio"] >= 0.9 else ("moderately stable" if stability["stillness_ratio"] >= 0.7 else "restless")
        motion_level = label_intensity(motion["movement_intensity_avg"]).replace("very much", "very high").replace("relatively many", "high")
        space_level = label_ratio(space_use["bbox_area_avg"]).replace("very much", "very large").replace("relatively many", "large").replace("medium", "moderate").replace("little", "small").replace("none", "very small")
        feedback = {
            "posture_feedback": f"Posture is {posture_level}.",
            "stability_feedback": f"Body stability: {stability_level}.",
            "motion_feedback": f"Movement intensity is {motion_level}.",
            "space_feedback": f"Framing/body space usage is {space_level}.",
        }
        return {
            "posture": posture,
            "stability": stability,
            "space_use": space_use,
            "motion": motion,
            "feedback": feedback,
        }


# -------- Gesture analyzer (MediaPipe Tasks) --------
class GestureAnalyzer(Analyzer):
    name: str = "gesture"

    def __init__(self) -> None:
        self.detector = None
        self.dt_s = 0.0
        self.total_time_s = 0.0
        # per-class stats
        self.class_stats: Dict[str, Dict[str, float]] = {}
        # hand visibility durations
        self.left_vis_s = 0.0
        self.right_vis_s = 0.0
        # prev top class per hand for event counts
        self.prev_class = {"left": None, "right": None}
        self.conf_threshold = 0.5

    def start(self, video_props: Dict[str, Any]) -> None:
        model_path = ensure_model_from(GESTURE_MODEL_URL, GESTURE_MODEL_LOCAL)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            num_hands=2,
        )
        self.detector = mp_vision.GestureRecognizer.create_from_options(options)
        ms_per_frame = float(video_props.get("ms_per_frame", 33.3))
        sample_n = int(video_props.get("sample_n", 1))
        self.dt_s = (ms_per_frame * max(1, sample_n)) / 1000.0

    def _bump_class(self, cls: str, inc_time: bool, inc_count: bool) -> None:
        if cls is None:
            return
        entry = self.class_stats.setdefault(cls, {"duration_s": 0.0, "count": 0})
        if inc_time:
            entry["duration_s"] += self.dt_s
        if inc_count:
            entry["count"] += 1

    def process(self, frame_rgb, frame_index: int, timestamp_ms: int) -> Optional[Dict[str, Any]]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self.detector.recognize_for_video(mp_image, timestamp_ms)

        # Results may contain gestures and handedness per hand index
        # Determine top class per hand
        hand_top = []  # list of tuples: (handedness, class_name or None)
        if res and getattr(res, "gestures", None):
            # res.gestures: List[List[Category]] per hand
            # res.handedness: List[List[Category]] per hand
            for i, cats in enumerate(res.gestures):
                handed = None
                if getattr(res, "handedness", None) and len(res.handedness) > i and len(res.handedness[i]) > 0:
                    handed = res.handedness[i][0].category_name.lower()  # 'left' or 'right'
                top_cls = None
                if cats and len(cats) > 0:
                    if cats[0].score >= self.conf_threshold:
                        top_cls = cats[0].category_name
                hand_top.append((handed, top_cls))

        # Update visibility and class stats
        seen_sides = set()
        for handed, top_cls in hand_top:
            if handed in ("left", "right"):
                seen_sides.add(handed)
                # visibility
                if handed == "left":
                    self.left_vis_s += self.dt_s
                else:
                    self.right_vis_s += self.dt_s
                # class duration
                if top_cls is not None:
                    self._bump_class(top_cls, inc_time=True, inc_count=False)
                # event count on change
                prev = self.prev_class.get(handed)
                if top_cls is not None and top_cls != prev:
                    self._bump_class(top_cls, inc_time=False, inc_count=True)
                self.prev_class[handed] = top_cls

        self.total_time_s += self.dt_s
        # No per-frame payload currently
        return None

    def finalize(self) -> Dict[str, Any]:
        total = max(self.total_time_s, 1e-9)
        any_vis = (self.left_vis_s + self.right_vis_s - min(self.left_vis_s, self.right_vis_s))  # upper bound; approx
        hand_visibility = {
            "left_ratio": min(1.0, self.left_vis_s / total),
            "right_ratio": min(1.0, self.right_vis_s / total),
            "any_ratio": min(1.0, max(self.left_vis_s, self.right_vis_s) / total),
        }
        dominant = "none"
        if self.left_vis_s > self.right_vis_s:
            dominant = "left"
        elif self.right_vis_s > self.left_vis_s:
            dominant = "right"
        # Class frequency feedback: sum counts across classes
        total_gesture_events = sum(int(v.get("count", 0)) for v in self.class_stats.values())
        class_level = label_count(total_gesture_events)
        visibility_level = label_ratio(hand_visibility["any_ratio"]).replace("very much", "very high").replace("relatively many", "high").replace("medium", "moderate")
        feedback = {
            "gesture_frequency_feedback": f"Gesture frequency is {class_level}.",
            "hand_visibility_feedback": f"Hands visibility is {visibility_level}.",
            "dominant_hand_feedback": f"Dominant hand: {dominant}.",
        }
        return {
            "gestureClasses": self.class_stats,
            "handVisibility": hand_visibility,
            "dominantHand": dominant,
            "feedback": feedback,
        }


# -------- Pipeline (decode once, fan out to modules) --------
def run_pipeline(local_video_path: str, sample_n: int, max_frames: Optional[int], modules: List[str]) -> Dict[str, Any]:
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ms_per_frame = 1000.0 / fps if fps > 0 else 33.3

    # Instantiate analyzers based on requested modules
    analyzers: Dict[str, Analyzer] = {}
    unsupported: List[str] = []
    for m in modules:
        if m == "hands":
            analyzers[m] = HandsAnalyzer()
        elif m == "face":
            analyzers[m] = FaceAnalyzer()
        elif m == "pose":
            analyzers[m] = PoseAnalyzer()
        elif m == "gesture":
            analyzers[m] = GestureAnalyzer()
        else:
            unsupported.append(m)

    video_props = {"fps": float(fps), "ms_per_frame": ms_per_frame, "sample_n": sample_n}
    for a in analyzers.values():
        a.start(video_props)

    frames = []
    frame_idx = 0
    sampled = 0

    # For cross-module metrics (e.g., self-touch), track latest landmarks
    last_pose_lm = None
    last_face_lm = None  # only nose needed

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % max(1, sample_n) != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            ts_ms = int(frame_idx * ms_per_frame)

            # Per-frame module outputs
            modules_out: Dict[str, Any] = {}
            for name, analyzer in analyzers.items():
                out = analyzer.process(frame_rgb, frame_idx, ts_ms)
                if out is not None:
                    modules_out[name] = out

            # Capture minimal pose/face landmarks for cross-module signals
            if "pose" in modules_out and "landmarks" in modules_out["pose"]:
                last_pose_lm = modules_out["pose"]["landmarks"]
            if "face" in analyzers:
                # Face analyzer doesn't output per-frame landmarks; try to get nose via Pose if available
                pass

            frames.append({
                "frame_index": frame_idx,
                "timestamp_ms": ts_ms,
                "modules": modules_out
            })

            sampled += 1
            frame_idx += 1
            if max_frames and sampled >= max_frames:
                break
    finally:
        cap.release()

    summaries = {name: analyzer.finalize() for name, analyzer in analyzers.items()}

    # -------- Cross-module: self-touch (hand near face) --------
    # Use pose wrists (15,16) and pose nose (0) if pose available.
    # Distance threshold: 0.08 in normalized image coords (heuristic), rising-edge count.
    def compute_self_touch(frames_list: List[Dict[str, Any]], dt_s: float) -> Dict[str, Any]:
        import math
        active = False
        duration = 0.0
        count = 0
        for fr in frames_list:
            pose = fr.get("modules", {}).get("pose")
            if not pose or "landmarks" not in pose or not pose["landmarks"]:
                # treat as inactive this sample
                active = False
                continue
            lm = pose["landmarks"]
            try:
                nose = lm[0]; lw = lm[15]; rw = lm[16]
                # normalized 2D distance to nearest wrist
                d_l = math.hypot(lw["x"] - nose["x"], lw["y"] - nose["y"]) if lw else 1e9
                d_r = math.hypot(rw["x"] - nose["x"], rw["y"] - nose["y"]) if rw else 1e9
                d = min(d_l, d_r)
            except Exception:
                d = 1e9
            threshold = 0.08
            if d <= threshold:
                if not active:
                    active = True
                    count += 1
                duration += dt_s
            else:
                active = False
        return {"duration_s": duration, "count": count}

    # dt_s is consistent across analyzers (same sampling); infer from pose or face analyzer if present
    dt_s = None
    if "pose" in analyzers and hasattr(analyzers["pose"], "dt_s"):
        dt_s = analyzers["pose"].dt_s
    elif "face" in analyzers and hasattr(analyzers["face"], "dt_s"):
        dt_s = analyzers["face"].dt_s
    if dt_s:
        self_touch = compute_self_touch(frames, dt_s)
        # Attach into gesture summary if gesture analyzer present; else create a lightweight entry
        if "gesture" in summaries:
            summaries["gesture"]["selfTouch"] = self_touch
        else:
            summaries["gesture"] = {"selfTouch": self_touch}

    return {
        "video": {
            "fps": float(fps),
            "sample_every_n_frames": sample_n,
            "frames_sampled": len(frames),
        },
        "frames": frames,
        "summaries": summaries,
        "unsupported_modules": unsupported
    }


# -------- Single entry route "/" --------
@app.post("/")
async def analyze_nonverbal(req: NonverbalRequest):
    modules = req.modules or ["hands"]
    local_video = to_local_file(req.videoUrl)

    try:
        # Run the CPU-bound pipeline in a worker thread to avoid blocking the event loop
        result = await to_thread.run_sync(
            run_pipeline,
            local_video,
            req.sampleEveryNFrames,
            req.maxFrames,
            modules
        )
        return result
    finally:
        # Cleanup temp file if downloaded, if needed
        try:
            if os.path.dirname(local_video) == tempfile.gettempdir():
                os.remove(local_video)
        except Exception:
            pass


# -------------- Local CLI runner (does not affect server) --------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local nonverbal hands analysis test runner")
    # Default to repo_root/test/test(2).mp4
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_video = os.path.join(repo_root, "test", "test(2).mp4")
    parser.add_argument("--path", dest="path", default=default_video, help="Local video path or HTTPS URL")
    parser.add_argument("--sample-n", dest="sample_n", type=int, default=10, help="Sample every N frames")
    parser.add_argument("--max-frames", dest="max_frames", type=int, default=5, help="Max sampled frames to process")
    parser.add_argument("--modules", nargs="*", default=["hands"], help="Modules to run (default: hands)")

    args = parser.parse_args()

    # Allow URL or local path
    local_video = to_local_file(args.path)
    try:
        result = run_pipeline(local_video, args.sample_n, args.max_frames, args.modules)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        # cleanup temp download if any
        try:
            if os.path.dirname(local_video) == tempfile.gettempdir():
                os.remove(local_video)
        except Exception:
            pass


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import os
# import tempfile
# import requests
# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python as mp_python
# from mediapipe.tasks.python import vision as mp_vision

# app = FastAPI()

# # Hand Landmarker model (download on first use)
# MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
# MODEL_LOCAL = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# class VideoPayload(BaseModel):
#     videoUrl: str



# @app.post("/")
# async def root(payload: VideoPayload):
#     print(payload.videoUrl)
#     return payload.videoUrl

# class NonVerbal:
    