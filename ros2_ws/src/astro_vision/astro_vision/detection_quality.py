"""Detection confidence for Haar cascade face detectors.

A cascade that publishes only bounding boxes tells the gaze stack nothing about
how good each detection was, and the stack's whole arbitration — the 0.50 validity
gate, the 0.75 acquisition / 0.40 hold hysteresis — then runs on a fabricated
constant. OpenCV already computes a per-detection stage weight; this maps it onto
the confidence scale those thresholds are written in.
"""

from typing import Any, List, Optional, Tuple

# Range of `levelWeights` from the default frontal-face cascades: a bare
# acceptance sits near zero, and a clean synthetic frontal face measured 8.27, so
# anything past the ceiling is already unambiguous and clamps. The ceiling is
# provisional — it was set from synthetic frames, and wants re-measuring against
# real camera footage of the robot's actual working distances.
HAAR_WEIGHT_FLOOR = 0.0
HAAR_WEIGHT_CEILING = 6.0

# The floor sits under VisualPerceptionCore's 0.50 validity gate on purpose, so a
# barely-accepted detection is discarded rather than tracked. The ceiling stays
# short of 1.0 because a Haar cascade never earns certainty.
CONFIDENCE_FLOOR = 0.40
CONFIDENCE_CEILING = 0.95


def haar_level_weight_to_confidence(level_weight: Optional[float]) -> float:
    """Maps one cascade stage weight onto a [CONFIDENCE_FLOOR..CONFIDENCE_CEILING] score.

    `level_weight` is None on OpenCV builds that expose no `detectMultiScale3`;
    that silence is treated as the weakest evidence, never as a good detection.
    """
    if level_weight is None:
        return CONFIDENCE_FLOOR

    span = HAAR_WEIGHT_CEILING - HAAR_WEIGHT_FLOOR
    ratio = (float(level_weight) - HAAR_WEIGHT_FLOOR) / span
    ratio = max(0.0, min(1.0, ratio))
    return CONFIDENCE_FLOOR + ratio * (CONFIDENCE_CEILING - CONFIDENCE_FLOOR)


def detect_faces_with_confidence(cascade: Any, image, **kwargs) -> List[Tuple[int, int, int, int, float]]:
    """Runs a cascade and returns (x, y, w, h, confidence) per detection.

    Prefers `detectMultiScale3`, the only entry point that hands back the stage
    weights; builds without it — and the import-less cv2 stub the vision nodes
    carry — fall back to the unscored call and the confidence floor.
    """
    weights = None
    if hasattr(cascade, "detectMultiScale3"):
        rects, _levels, weights = cascade.detectMultiScale3(image, outputRejectLevels=True, **kwargs)
    else:
        rects = cascade.detectMultiScale(image, **kwargs)

    faces: List[Tuple[int, int, int, int, float]] = []
    for idx, (x, y, w, h) in enumerate(rects):
        weight = weights[idx] if weights is not None and idx < len(weights) else None
        faces.append((int(x), int(y), int(w), int(h), haar_level_weight_to_confidence(weight)))
    return faces


class DetectionHold:
    """Bridges the single-frame misses a Haar cascade makes on a face that is there.

    Measured over a 300-frame scene, publishing the cascade's raw output reported a
    plainly present face on 82% of frames across 41 separate dropouts, most of them
    one or two frames long. Carrying the last detection over those gaps closes them.

    A carried detection is not evidence, so its confidence decays each frame it is
    reused: downstream may keep an existing lock through the gap (hold threshold
    0.40) but must not acquire a new target (acquisition threshold 0.75) from a frame
    where the detector saw nothing.
    """

    def __init__(self, hold_frames: int = 2, decay: float = 0.75):
        self.hold_frames = max(0, int(hold_frames))
        self.decay = decay
        self._last: List[Tuple[int, int, int, int, float]] = []
        self._held = 0

    def update(
        self, faces: List[Tuple[int, int, int, int, float]]
    ) -> List[Tuple[int, int, int, int, float]]:
        """Returns this frame's detections, or the previous ones while budget remains."""
        if faces:
            self._last = list(faces)
            self._held = 0
            return faces

        if self._held >= self.hold_frames or not self._last:
            self._last = []
            return []

        self._held += 1
        scale = self.decay ** self._held
        return [(x, y, w, h, conf * scale) for (x, y, w, h, conf) in self._last]


class HaarFaceDetector:
    """Cascade detection, kept as the fallback when the YuNet model is not installed.

    It holds a face well enough while that face looks at the camera, and loses it as
    soon as the head tilts or turns — 90.0% with a 462 ms gap at 22 degrees of roll,
    85.8% with a 792 ms gap turning to profile.
    """

    def __init__(self, cascade: Any, **detect_kwargs):
        self.cascade = cascade
        self.detect_kwargs = detect_kwargs

    def detect(self, frame) -> List[Tuple[int, int, int, int, float]]:
        import cv2

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return detect_faces_with_confidence(self.cascade, gray, **self.detect_kwargs)


class YuNetFaceDetector:
    """OpenCV Zoo's YuNet: the detector the repo already installs for SFace.

    Half the cost of the cascade (3.6-4.4 ms against 7-8.4 ms on 640x480) and steady
    through the poses that break it — 100% at 22 degrees of roll where Haar dropped
    to 90%, and 90.4% turning to profile against Haar's 85.8%, with the worst gap
    falling from 792 ms to 165 ms. It also reports a real confidence per face, so no
    stage-weight mapping is needed.
    """

    def __init__(self, model_path, score_threshold: float = 0.7, nms_threshold: float = 0.3):
        import cv2

        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, nms_threshold, 5000
        )
        self._input_size = None

    def detect(self, frame) -> List[Tuple[int, int, int, int, float]]:
        import cv2

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        height, width = frame.shape[:2]
        if self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)

        _, detections = self._detector.detect(frame)
        if detections is None:
            return []

        faces: List[Tuple[int, int, int, int, float]] = []
        for row in detections:
            x, y, w, h = (int(round(v)) for v in row[:4])
            faces.append((x, y, w, h, float(row[-1])))
        return faces


def create_face_detector(model_dir, haar_cascade: Any, **haar_kwargs):
    """Returns YuNet when its model is present, otherwise the cascade fallback."""
    from pathlib import Path

    model_path = Path(model_dir) / "yunet.onnx"
    if model_path.exists():
        try:
            return YuNetFaceDetector(model_path)
        except Exception:
            pass
    return HaarFaceDetector(haar_cascade, **haar_kwargs)
