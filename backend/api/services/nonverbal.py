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
import spacy
from collections import Counter
import nltk
from nltk.corpus import stopwords
import random

nltk.download('stopwords')

app = FastAPI()
# Safe spaCy model loading (optional)
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

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


def label_blink_count(x: float) -> str:
    """Blink-specific qualitative labels with higher thresholds.
    Edges tuned to be more forgiving; only very high blink counts map to 'very much'.
    Bins: (0,10) little, [10,25) medium, [25,70) relatively many, >=70 very much
    """
    return label_by_edges(x, [10.0, 25.0, 70.0])


# -------- Scoring helpers (0-100) --------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def score_linear(x: float, lo: float, hi: float, min_score: int = 20, max_score: int = 95) -> int:
    """Increasing score: lo -> min_score, hi -> max_score."""
    try:
        x = float(x)
    except Exception:
        return min_score
    if hi == lo:
        return int((min_score + max_score) / 2)
    t = _clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return int(round(min_score + t * (max_score - min_score)))

def score_inverse(x: float, lo: float, hi: float, min_score: int = 20, max_score: int = 95) -> int:
    """Decreasing score: x <= lo -> max_score, x >= hi -> min_score."""
    try:
        x = float(x)
    except Exception:
        return min_score
    if hi == lo:
        return int((min_score + max_score) / 2)
    t = _clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return int(round(max_score - t * (max_score - min_score)))

def score_triangular(x: float, low: float, mid: float, high: float, min_score: int = 20, max_score: int = 95) -> int:
    """Peak at mid; declines linearly toward low/high; outside gives min_score."""
    try:
        x = float(x)
    except Exception:
        return min_score
    if x <= low or x >= high:
        return min_score
    if x == mid:
        return max_score
    if x < mid:
        t = (x - low) / max(1e-9, (mid - low))
    else:
        t = (high - x) / max(1e-9, (high - mid))
    t = _clamp(t, 0.0, 1.0)
    return int(round(min_score + t * (max_score - min_score)))


def score_trapezoid(
    x: float,
    low: float,
    plateau_lo: float,
    plateau_hi: float,
    high: float,
    min_score: int = 20,
    max_score: int = 95,
) -> int:
    """Plateau-shaped score with a flat optimal region [plateau_lo, plateau_hi].

    - x <= low or x >= high -> min_score
    - low < x < plateau_lo: linearly rises from min_score to max_score
    - plateau_lo <= x <= plateau_hi: max_score
    - plateau_hi < x < high: linearly falls from max_score to min_score
    """
    try:
        x = float(x)
    except Exception:
        return min_score
    # Guard invalid configuration
    if not (low <= plateau_lo <= plateau_hi <= high) or (low == high):
        return int((min_score + max_score) / 2)

    if x <= low or x >= high:
        return min_score
    if plateau_lo <= x <= plateau_hi:
        return max_score
    if x < plateau_lo:
        t = (x - low) / max(1e-9, (plateau_lo - low))
        t = _clamp(t, 0.0, 1.0)
        return int(round(min_score + t * (max_score - min_score)))
    # x > plateau_hi
    t = (high - x) / max(1e-9, (high - plateau_hi))
    t = _clamp(t, 0.0, 1.0)
    return int(round(min_score + t * (max_score - min_score)))


# -------- Score post-processing: lift low scores --------
def lift_low_score(score: int) -> int:
    """Map scores below 60 into [50, 60) while keeping >=60 unchanged.

    Examples:
    - 30 -> ~55
    - 20 -> ~53
    - 0  -> 50
    - 69, 73, 91 stay the same.
    """
    try:
        s = float(score)
    except Exception:
        return 50
    if s >= 60.0:
        return int(round(s))
    s = max(0.0, s)
    return int(round(50.0 + (s * (10.0 / 60.0))))


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
            "blinking_feedback": f"Blinking frequency: {label_blink_count(blink_count)}",
        }

        # Scoring (0-100; ~70 as typical)
        speaking_ratio = jaw['active_duration_s'] / max(self.total_time_s, 1e-9)
        # Smile: moderate smiles are best; very low or very high intensity scores lower
        # Plateau roughly between ~0.015 and ~0.05 (blendshape average), then gently decreases
        smile_score = score_trapezoid(smile['avg'], low=0.0, plateau_lo=0.015, plateau_hi=0.05, high=0.20)
        # Speaking: increasing with ratio (typical ~0.15 => ~70)
        speaking_score = score_linear(speaking_ratio, lo=0.05, hi=0.30)
        # Blinking: be more forgiving; only very high blink counts should score low.
        # Use an inverse curve: <= ~25 stays high, drops gradually towards ~70+ blinks.
        blink_score = score_inverse(blink_count, lo=25, hi=70)
        scores = {
            "smile_score": lift_low_score(smile_score),
            "speaking_score": lift_low_score(speaking_score),
            "blinking_score": lift_low_score(blink_score),
        }

        return {
            "smile": smile,
            "jawOpen": jaw,
            "eyeBlinkLeft": ebl,
            "eyeBlinkRight": ebr,
            "blinkCount": blink_count,
            "feedback": feedback,
            "scores": scores,
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
        # Return pose data directly (no extra nesting) so downstream code can access keys like 'landmarks'
        return out

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
        # Scoring
        # Posture: higher avg tilt_deg means closer to horizontal shoulder line => more upright in our current coding
        posture_score = score_linear(posture["tilt_deg_avg"], lo=150.0, hi=178.0)
        # Stability: higher stillness better up to 0.95; then plateau
        stability_score = score_linear(stability["stillness_ratio"], lo=0.6, hi=0.95)
        # Motion: best around moderate intensity (triangular)
        motion_score = score_triangular(motion["movement_intensity_avg"], low=0.0, mid=0.12, high=0.35)
        # Space: prefer a plateau of average bbox area between ~30% and ~50% of frame
        # Below ~10% or above ~70% is poor; 10%-30% ramps up; 50%-70% ramps down
        space_score = score_trapezoid(space_use["bbox_area_avg"], low=0.10, plateau_lo=0.30, plateau_hi=0.50, high=0.70)
        scores = {
            "posture_score": lift_low_score(posture_score),
            "stability_score": lift_low_score(stability_score),
            "motion_score": lift_low_score(motion_score),
            "space_score": lift_low_score(space_score),
        }
        return {
            "posture": posture,
            "stability": stability,
            "space_use": space_use,
            "motion": motion,
            "feedback": feedback,
            "scores": scores,
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
        # Visibility feedback only (remove gesture frequency and dominant-hand feedback)
        visibility_level = label_ratio(hand_visibility["any_ratio"]).replace("very much", "very high").replace("relatively many", "high").replace("medium", "moderate")
        feedback = {
            "hand_visibility_feedback": f"Hands visibility is {visibility_level}.",
        }
        # Scores (keep visibility; selfTouch added later in pipeline)
        # Visibility: more visible hands up to ~0.7 is good
        visibility_score = score_linear(hand_visibility["any_ratio"], lo=0.2, hi=0.7)
        scores = {
            "hand_visibility_score": lift_low_score(visibility_score),
        }
        return {
            "gestureClasses": self.class_stats,
            "handVisibility": hand_visibility,
            "dominantHand": dominant,
            "feedback": feedback,
            "scores": scores,
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
                # Consider multiple face-adjacent points (MP Pose indices):
                # nose(0), eyes(1-6), ears(7-8), mouth corners(9-10)
                face_idxs = [0,1,2,3,4,5,6,7,8,9,10]
                face_pts = []
                for idx in face_idxs:
                    try:
                        p = lm[idx]
                        if p is not None:
                            face_pts.append((float(p["x"]), float(p["y"])) )
                    except Exception:
                        continue
                # Wrist points
                lw = lm[15] if len(lm) > 15 else None
                rw = lm[16] if len(lm) > 16 else None
                def min_dist_to_face(wrist):
                    if wrist is None or not face_pts:
                        return 1e9
                    wx, wy = float(wrist.get("x", 1e9)), float(wrist.get("y", 1e9))
                    md = 1e9
                    for fx, fy in face_pts:
                        d = math.hypot(wx - fx, wy - fy)
                        if d < md:
                            md = d
                    return md
                d_l = min_dist_to_face(lw)
                d_r = min_dist_to_face(rw)
                d = min(d_l, d_r)
            except Exception:
                d = 1e9
            # Threshold tuned for normalized image coords; raised to better catch forehead touches
            threshold = 0.15
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
            # Add selfTouch feedback and score (less is better)
            st = self_touch
            st_score = score_inverse(st.get("duration_s", 0.0), lo=0.0, hi=2.0)
            st_level = label_ratio(min(1.0, st.get("duration_s", 0.0) / max(1e-9, (len(frames) * (dt_s))))).replace("very much", "very high").replace("relatively many", "high").replace("medium", "moderate")
            summaries["gesture"].setdefault("feedback", {})["self_touch_feedback"] = f"Self-touch is {('low' if st_score>70 else 'moderate' if st_score>50 else 'high')}."
            summaries["gesture"].setdefault("scores", {})["self_touch_score"] = lift_low_score(st_score)
        else:
            summaries["gesture"] = {"selfTouch": self_touch}

    # -------- Aggregate simplified scores and remove original per-module scores --------
    # New scores to return (only these):
    # - facial_expression_score (from face.smile_score)
    # - eye_movements_score (from face.blinking_score)
    # - pausing_score (from face.speaking_score)
    # - posture_score (average of pose.stability_score and pose.posture_score)
    # - hand_gesture_score (weighted: (hand_visibility_score + 2*self_touch_score) / 3)
    # - spatial_distribution_score (from pose.space_score)
    aggregated_scores: Dict[str, int] = {}

    # Helper to safely fetch nested score
    def _get_score(section: str, key: str) -> Optional[int]:
        sect = summaries.get(section)
        if not sect:
            return None
        sc = sect.get("scores")
        if not sc:
            return None
        val = sc.get(key)
        try:
            return int(val) if val is not None else None
        except Exception:
            return None

    # Face-derived scores
    face_smile = _get_score("face", "smile_score")
    face_blink = _get_score("face", "blinking_score")
    face_speaking = _get_score("face", "speaking_score")
    if face_smile is not None:
        aggregated_scores["facial_expression_score"] = face_smile
    if face_blink is not None:
        aggregated_scores["eye_movements_score"] = face_blink
    if face_speaking is not None:
        # Pausing should track speaking richness (more speaking -> higher pausing_score)
        aggregated_scores["pausing_score"] = face_speaking

    # Pose-derived scores
    pose_posture = _get_score("pose", "posture_score")
    pose_stability = _get_score("pose", "stability_score")
    pose_space = _get_score("pose", "space_score")
    if pose_posture is not None and pose_stability is not None:
        aggregated_scores["posture_score"] = int(round((pose_posture + pose_stability) / 2))
    if pose_space is not None:
        aggregated_scores["spatial_distribution_score"] = pose_space

    # Gesture-derived scores
    gest_vis = _get_score("gesture", "hand_visibility_score")
    gest_self_touch = _get_score("gesture", "self_touch_score")
    if gest_vis is not None and gest_self_touch is not None:
        # Weighted: self_touch has double weight compared to visibility
        aggregated_scores["hand_gesture_score"] = int(round((gest_vis + 2 * gest_self_touch) / 3))

    # -------- Build data-driven descriptions (positives & negatives) from 9 standards --------
    # Collect the original nine standards (when available)
    standard_scores: Dict[str, int] = {}
    # Face
    f_smile = _get_score("face", "smile_score")
    f_speaking = _get_score("face", "speaking_score")
    f_blink = _get_score("face", "blinking_score")
    if f_smile is not None:
        standard_scores["smile"] = f_smile
    if f_speaking is not None:
        standard_scores["speaking"] = f_speaking
    if f_blink is not None:
        standard_scores["blinking"] = f_blink
    # Pose
    p_posture = _get_score("pose", "posture_score")
    p_stability = _get_score("pose", "stability_score")
    p_motion = _get_score("pose", "motion_score")
    p_space = _get_score("pose", "space_score")
    if p_posture is not None:
        standard_scores["posture"] = p_posture
    if p_stability is not None:
        standard_scores["stability"] = p_stability
    if p_motion is not None:
        standard_scores["motion"] = p_motion
    if p_space is not None:
        standard_scores["space"] = p_space
    # Gesture
    if gest_vis is not None:
        standard_scores["hand_visibility"] = gest_vis
    if gest_self_touch is not None:
        standard_scores["self_touch"] = gest_self_touch

    # ---- Debug print: nine original standards (not returned) ----
    try:
        all_keys = [
            "smile",
            "speaking",
            "blinking",
            "posture",
            "stability",
            "motion",
            "space",
            "hand_visibility",
            "self_touch",
        ]
        debug_payload = {k: standard_scores.get(k, "N/A") for k in all_keys}
        print("[nonverbal] Nine standard scores:", json.dumps(debug_payload, ensure_ascii=False))
    except Exception as e:
        print("[nonverbal] Failed to print nine standard scores:", e)

    # Description templates: 4 positives and 4 negatives per standard
    POS_DESC: Dict[str, List[str]] = {
        "smile": [
            "Smiles appear natural and welcoming.",
            "Friendly facial expression comes through.",
            "Warmth is conveyed with appropriate smiling.",
            "Pleasant expression helps build rapport.",
        ],
        "speaking": [
            "Speaking presence feels lively and clear.",
            "Mouth movement shows confident articulation.",
            "Vocal delivery maintains steady energy.",
            "Good verbal flow supports the message.",
        ],
        "blinking": [
            "Blinking rate appears comfortable and natural.",
            "Eye behavior feels relaxed and steady.",
            "No signs of excessive blinking.",
            "Eye movements support an attentive look.",
        ],
        "posture": [
            "Posture meets the standard and is reasonable.",
            "Shoulders look level; stance appears composed.",
            "Body alignment looks professional.",
            "Upright presence supports credibility.",
        ],
        "stability": [
            "Body remains steady without fidgeting.",
            "Minimal sway keeps attention on the message.",
            "Grounded stance shows control.",
            "Stable presence feels confident.",
        ],
        "motion": [
            "Gesture and movement feel purposeful.",
            "Moderate movement adds energy.",
            "Dynamic range supports emphasis.",
            "Expressive motion helps engagement.",
        ],
        "space": [
            "Framing and body space feel balanced.",
            "Uses space comfortably within the frame.",
            "Presence fills the frame appropriately.",
            "Composition supports a professional look.",
        ],
        "hand_visibility": [
            "Hands are visible and support communication.",
            "Open-hand visibility aids clarity.",
            "Gestures are easy to see.",
            "Hand presence reinforces key points.",
        ],
        "self_touch": [
            "Minimal self-touch keeps focus on the message.",
            "Looks composed with few face touches.",
            "Clean delivery without self-soothing cues.",
            "Professional poise with controlled touches.",
        ],
    }
    NEG_DESC: Dict[str, List[str]] = {
        "smile": [
            "Expression looks flat; consider a bit more warmth.",
            "Limited smiling reduces approachability.",
            "Facial affect feels restrained.",
            "Add brief, natural smiles to connect.",
        ],
        "speaking": [
            "Long stretches of silence; aim for steadier delivery.",
            "Mouth activity suggests low projection.",
            "Speaking presence feels subdued.",
            "Vocal energy dips; increase engagement.",
        ],
        "blinking": [
            "Blinking is frequent; try relaxing the gaze.",
            "Eye behavior may distract at times.",
            "Rapid blinking suggests tension.",
            "Consider a steadier eye rhythm.",
        ],
        "posture": [
            "Posture tilts at times; aim for a steadier stance.",
            "Shoulder tilt reduces presence.",
            "Body alignment looks uneven.",
            "Straighter posture would read more confident.",
        ],
        "stability": [
            "Noticeable sway or fidgeting draws attention.",
            "Movement noise competes with content.",
            "Restlessness reduces clarity.",
            "Try anchoring and easing micro-movements.",
        ],
        "motion": [
            "Movement intensity varies; consider moderating.",
            "Gestures feel sparse or restless at times.",
            "Motion occasionally distracts from the message.",
            "A steadier movement baseline would help clarity.",
        ],
        "space": [
            "Appears cramped or drifts out of frame.",
            "Space use feels uneven.",
            "Presence is either too tight or too loose.",
            "Recenter and maintain consistent framing.",
        ],
        "hand_visibility": [
            "Hands rarely appear; bring them into frame.",
            "Low hand visibility reduces expressiveness.",
            "Hidden hands limit gesture impact.",
            "Show purposeful hands near mid-frame.",
        ],
        "self_touch": [
            "Frequent self-touch draws attention.",
            "Face-touch habits can signal tension.",
            "Self-soothing gestures appear often.",
            "Reduce face touches to avoid distraction.",
        ],
    }

    # Direction-aware negative templates for selected standards
    NEG_DESC_DIR: Dict[str, Dict[str, List[str]]] = {
        "smile": {
            # Too little smiling
            "low": [
                "Expression looks flat; add brief, natural smiles.",
                "Limited smiling reduces approachability; show a touch more warmth.",
                "Facial affect feels restrained; sprinkle in light smiles.",
                "A few natural smiles would help you connect.",
            ],
            # Too much smiling
            "high": [
                "Smiling appears exaggerated at times; ease it slightly.",
                "Strong smiles can feel constant; relax the intensity a bit.",
                "Frequent or large smiles may distract from the message; soften occasionally.",
                "Dial back the smile intensity to keep it natural.",
            ],
        },
        "motion": {
            # Movement too low
            "low": [
                "Movement is minimal; add a few purposeful gestures.",
                "Gestures are sparse; introduce moderate motion for emphasis.",
                "Very low movement can read as flat; animate key points lightly.",
                "Consider a touch more expressive motion to engage.",
            ],
            # Movement too high
            "high": [
                "Excessive movement distracts; reduce intensity and hold beats.",
                "Gestures run high; simplify and pause to highlight ideas.",
                "Rapid motion competes with content; slow down and ground gestures.",
                "Calm the overall movement for clearer delivery.",
            ],
        },
        "space": {
            # Framing too tight (too little space)
            "low": [
                "Framing is too tight; step back or widen the crop.",
                "Body occupies too little frame space; increase distance slightly.",
                "Composition feels cramped; add margin around shoulders and hands.",
                "Widen the view to reach a comfortable 30–50% frame use.",
            ],
            # Framing too loose (too much space)
            "high": [
                "Framing is too loose; move closer or crop tighter.",
                "Presence gets small in frame; reduce empty space.",
                "Wide composition dilutes impact; tighten to center focus.",
                "Aim for a balanced 30–50% frame use for presence.",
            ],
        },
    }

    positive_texts: List[str] = []
    negative_texts: List[str] = []
    items = list(standard_scores.items())  # List[(key, score)]
    if items:
        # Positives selection
        high_keys = [k for k, s in items if s > 85]
        pos_selected: List[str] = []
        if len(high_keys) >= 3:
            # Randomly choose 3 from all >85 (not necessarily the top)
            pos_selected = random.sample(high_keys, 3)
        else:
            # Choose highest-scoring two (ordered by score)
            top_sorted = sorted(items, key=lambda t: t[1], reverse=True)
            take = min(2, len(top_sorted))
            pos_selected = [k for k, _ in top_sorted[:take]]

        # Negatives selection (avoid overlapping with positives)
        low_candidates = [(k, s) for k, s in items if s < 80 and k not in pos_selected]
        if len(low_candidates) >= 3:
            neg_pool_keys = [k for k, _ in low_candidates]
            neg_selected = random.sample(neg_pool_keys, 3)
        else:
            # Choose lowest two overall, excluding those already picked as positives
            bottom_sorted = sorted([(k, s) for k, s in items if k not in pos_selected], key=lambda t: t[1])
            take = min(2, len(bottom_sorted))
            neg_selected = [k for k, _ in bottom_sorted[:take]]

        # Determine direction (low/high) for selected directional standards using raw metrics
        dir_map: Dict[str, Optional[str]] = {}
        try:
            # Smile direction from face analyzer average
            face_summary = summaries.get("face", {}) if isinstance(summaries.get("face", {}), dict) else {}
            smile_dict = face_summary.get("smile") if isinstance(face_summary.get("smile"), dict) else None
            if smile_dict:
                s_avg = float(smile_dict.get("avg", 0.0))
                if s_avg < 0.015:
                    dir_map["smile"] = "low"
                elif s_avg > 0.05:
                    dir_map["smile"] = "high"

            # Motion direction from pose analyzer movement_intensity_avg (mid ~0.12)
            pose_summary = summaries.get("pose", {}) if isinstance(summaries.get("pose", {}), dict) else {}
            motion_dict = pose_summary.get("motion") if isinstance(pose_summary.get("motion"), dict) else None
            if motion_dict:
                m_avg = float(motion_dict.get("movement_intensity_avg", 0.0))
                dir_map["motion"] = "low" if m_avg < 0.12 else "high"

            # Space direction from pose analyzer bbox_area_avg with plateau 0.30–0.50
            space_dict = pose_summary.get("space_use") if isinstance(pose_summary.get("space_use"), dict) else None
            if space_dict:
                a_avg = float(space_dict.get("bbox_area_avg", 0.0))
                if a_avg < 0.30:
                    dir_map["space"] = "low"
                elif a_avg > 0.50:
                    dir_map["space"] = "high"
        except Exception:
            pass

        # Map selections to one randomized sentence each
        for key in pos_selected:
            opts = POS_DESC.get(key, [])
            if opts:
                positive_texts.append(random.choice(opts))
        for key in neg_selected:
            opts: List[str] = []
            # Use direction-aware negatives for selected standards when direction is known
            if key in NEG_DESC_DIR:
                direction = dir_map.get(key)
                if direction:
                    opts = NEG_DESC_DIR[key].get(direction, [])
            # Fallback to generic negatives if no direction or not directional
            if not opts:
                opts = NEG_DESC.get(key, [])
            if opts:
                negative_texts.append(random.choice(opts))

    # Remove old per-module scores from summaries as requested
    for key in list(summaries.keys()):
        if isinstance(summaries.get(key), dict) and "scores" in summaries.get(key, {}):
            try:
                del summaries[key]["scores"]
            except Exception:
                pass

    # Build minimal return payload
    expected_keys = [
        "facial_expression_score",
        "eye_movements_score",
        "pausing_score",
        "posture_score",
        "spatial_distribution_score",
        "hand_gesture_score",
    ]
    # Ensure all six keys present; default to 0 if not computed (e.g., module not requested)
    sub_scores = {k: int(aggregated_scores.get(k, 0) or 0) for k in expected_keys}
    # Overall is the mean of the six sub-scores
    overall_score = int(round(sum(sub_scores.values()) / len(sub_scores))) if sub_scores else 0

    return {
        "sub_scores": sub_scores,
        "descriptions": {
            "positive": positive_texts,
            "negative": negative_texts,
        },
        "overall_score": overall_score,
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

class VerbalRequest(BaseModel):
    cleansedText: str

@app.post("/vocab-fillers")
async def analyze_nonverbal(req: VerbalRequest):
    if nlp is None:
        # Fallback: return the text as-is if spaCy model is unavailable
        return req.cleansedText
    doc = nlp(req.cleansedText)

    # Token-level analysis
    words = [token.text.lower() for token in doc if token.is_alpha]  # Only words
    unique_words = set(words)
    vocab_richness = round(len(unique_words) / len(words) * 100, 1)  # Higher = more diverse vocabulary

    word_counts = Counter(words)
    filler_words = [
        "um",
        "uh",
        "like",
        "and yeah",
        "so yeah",
        "you know",
        "so",
        "actually",
        "basically",
        "right",
        "i mean",
        "okay",
        "well",
        "yeah",
    ]
    filler_count = sum(word_counts[word] for word in filler_words if word in word_counts)

    return [vocab_richness, filler_count, len(words)]
