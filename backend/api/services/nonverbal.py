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
        else:
            unsupported.append(m)

    video_props = {"fps": float(fps), "ms_per_frame": ms_per_frame}
    for a in analyzers.values():
        a.start(video_props)

    frames = []
    frame_idx = 0
    sampled = 0

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
        # Cleanup temp file if downloaded
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
    