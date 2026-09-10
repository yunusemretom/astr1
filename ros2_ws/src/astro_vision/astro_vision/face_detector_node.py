#!/usr/bin/env python3
"""ASTRO V1 — Spatial AI Vision Node with Emotion & Gaze Tracking.

Features:
  1. 3D Head Pose & Gaze Angle (solvePnP / eye symmetry)
  2. Stereo Depth Distance (/vision/user_distance in meters)
  3. Facial Emotion Detection (/vision/user_emotion: 'happy', 'sad', 'surprised', 'neutral')
  4. Publishes:
      /vision/faces            (String JSON)
      /vision/person_detected  (Bool)
      /vision/looking_at_robot (Bool)
      /vision/head_yaw         (Float32)
      /vision/user_distance    (Float32)
      /vision/user_emotion     (String)
      /vision/face_image       (Image)
"""

import json
import logging
import os
import time

_LOG = logging.getLogger(__name__)

from collections import deque
try:
    import cv2
except ImportError:
    class _MockCV2:
        class data:
            haarcascades = ""
        class CascadeClassifier:
            def __init__(self, *args, **kwargs): pass
            def detectMultiScale(self, *args, **kwargs): return []
        INTER_AREA = 3
        FONT_HERSHEY_SIMPLEX = 0
        def resize(self, src, dsize, *args, **kwargs): return src
        def cvtColor(self, src, code): return src
        def rectangle(self, *args, **kwargs): pass
        def putText(self, *args, **kwargs): pass
    cv2 = _MockCV2()

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import String, Bool, Float32
except ImportError:
    try:
        from astro_vision.ros_compat import MockMsg, MockNode as Node, MockRclpy
    except ImportError:
        from ros_compat import MockMsg, MockNode as Node, MockRclpy
    rclpy = MockRclpy()
    qos_profile_sensor_data = 10
    Image = String = Bool = Float32 = MockMsg

try:
    from astro_vision.image_utils import bgr_to_imgmsg, imgmsg_to_bgr
    from astro_vision.face_recognizer import FaceRecognizer
    from astro_vision.detection_quality import create_face_detector
except ImportError:
    from image_utils import bgr_to_imgmsg, imgmsg_to_bgr
    from detection_quality import create_face_detector
    class FaceRecognizer:
        def identify(self, frame, x, y, w, h):
            return {"name": "Misafir", "title": "Ziyaretçi", "confidence": 0.0, "is_known": False}


class SpatialVisionNode(Node):
    def __init__(self):
        super().__init__("face_detector_node")
        self.declare_parameter("input_topic", "/oak/rgb/image_raw")
        self.declare_parameter("depth_topic", "/oak/stereo/image_raw")
        # Cascade pyramid tuning. Only reaches the Haar fallback: when yunet.onnx is
        # installed these have no effect, because YuNet does its own scale search.
        self.declare_parameter("scale_factor", 1.1)
        self.declare_parameter("min_neighbors", 4)
        self.declare_parameter("min_size", 45)
        self.declare_parameter("show_debug", False)
        # Process every frame. The old hardcoded `% 2` skip sat on top of a publisher
        # already halved to 15 Hz, leaving 7.5 Hz of bearings — one update per 133 ms —
        # while a loaded frame costs ~13.5 ms and the CPU idled through the rest.
        self.declare_parameter("process_every_n", 1)

        input_topic = self.get_parameter("input_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        self.scale_factor = float(self.get_parameter("scale_factor").value)
        self.min_neighbors = int(self.get_parameter("min_neighbors").value)
        self.min_size = int(self.get_parameter("min_size").value)
        self.show_debug = bool(self.get_parameter("show_debug").value)
        self.process_every_n = max(1, int(self.get_parameter("process_every_n").value))

        # Load Cascades (Default + Alt2 for maximum detection rate)
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        smile_path = cv2.data.haarcascades + "haarcascade_smile.xml"
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"

        self.face_cascade = cv2.CascadeClassifier(frontal_path)
        # YuNet is what scripts/install_face_models.sh already fetches for SFace, and
        # it beats the cascade on both counts: half the cost (3.6-4.4 ms against
        # 7-8.4 ms) and steady through head pose, where the cascade collapses — 100%
        # against 90.0% at 22 degrees of roll, and a worst gap of 165 ms against
        # 792 ms turning to profile. Detection was running on the weaker of two
        # detectors that were both already installed.
        model_dir = os.path.expanduser(os.getenv("FACE_MODEL_DIR", "~/.astro/models"))
        self.face_detector = create_face_detector(
            model_dir, self.face_cascade, **self._face_detect_kwargs()
        )
        self.get_logger().info(
            f"🔍 [Yüz algılama] {type(self.face_detector).__name__}"
        )
        self.smile_cascade = cv2.CascadeClassifier(smile_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)

        # Check GPU / CUDA Acceleration on Jetson
        self.gpu_accelerated = False
        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.gpu_accelerated = True
                self.get_logger().info(f"🚀 [Spatial Vision] Jetson Orin Nano GPU / CUDA Hızlandırma Aktif!")
        except Exception as _exc:
            self.get_logger().debug(f"__init__: yok sayılan hata ({_exc})")

        # Internal Buffers
        self._latest_depth = None
        self._gaze_history = deque(maxlen=5)
        self._emotion_history = deque(maxlen=8)
        self.face_recognizer = FaceRecognizer()

        # Telemetry counters
        self.invalid_roi_count: int = 0
        self.clipped_bbox_count: int = 0
        self.head_yaw_fallback_count: int = 0
        self.detector_exception_count: int = 0
        self.real_detections_count: int = 0
        self.synthetic_hold_detections_count: int = 0
        self._last_perf_log_time: float = time.time()

        # Publishers
        self.pub_faces = self.create_publisher(String, "/vision/faces", 10)
        self.pub_person = self.create_publisher(Bool, "/vision/person_detected", 10)
        self.pub_looking = self.create_publisher(Bool, "/vision/looking_at_robot", 10)
        self.pub_yaw = self.create_publisher(Float32, "/vision/head_yaw", 10)
        self.pub_distance = self.create_publisher(Float32, "/vision/user_distance", 10)
        self.pub_emotion = self.create_publisher(String, "/vision/user_emotion", 10)
        self.pub_recognized_person = self.create_publisher(String, "/vision/recognized_person", 10)
        self.pub_image = self.create_publisher(Image, "/vision/face_image", 10)

        # Subscribers
        # Görüntü akışları sensör QoS'u (BEST_EFFORT) kullanır: kare kaybı, geciken
        # kareler için retransmission yapmaktan iyidir. BEST_EFFORT abone RELIABLE
        # yayıncıdan da veri alabilir, bu yüzden depthai_ros_driver ile uyumludur.
        # Queue derinliği 1: yalnızca en son kare işlenir, eski kareler atılır → gecikme sıfıra yakın.
        try:
            from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
            latest_frame_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
        except ImportError:
            latest_frame_qos = 10
        self.sub_rgb = self.create_subscription(Image, input_topic, self.image_callback, latest_frame_qos)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.depth_callback, latest_frame_qos)

        self.get_logger().info(f"👁️ [Spatial Emotion Vision] 3D Bakış, Mesafe ve Yüz Duygu Analizi Aktif! RGB: {input_topic}")

    def depth_callback(self, msg: Image):
        try:
            if msg.encoding in ["16UC1", "mono16"]:
                self._latest_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            elif msg.encoding in ["32FC1"]:
                self._latest_depth = (np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width) * 1000.0).astype(np.uint16)
        except Exception as _exc:
            self.get_logger().debug(f"depth_callback: yok sayılan hata ({_exc})")

    def _estimate_distance(self, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> float:
        if self._latest_depth is not None:
            try:
                dh, dw = self._latest_depth.shape[:2]
                sx = int((x + w/2) * (dw / frame_w))
                sy = int((y + h/2) * (dh / frame_h))
                sx = max(0, min(dw - 1, sx))
                sy = max(0, min(dh - 1, sy))

                patch = self._latest_depth[max(0, sy - 10):min(dh, sy + 10), max(0, sx - 10):min(dw, sx + 10)]
                valid = patch[patch > 200]
                if len(valid) > 0:
                    return float(np.median(valid)) / 1000.0
            except Exception as _exc:
                self.get_logger().debug(f"_estimate_distance: yok sayılan hata ({_exc})")

        focal_length = frame_w * 0.8
        distance = (0.15 * focal_length) / max(1, w)
        return float(np.clip(distance, 0.3, 4.0))

    def _estimate_head_yaw(self, face_roi_gray, w, h) -> tuple[float, bool]:
        """Calculates yaw angle and strictly verifies eye visibility to reject side/back-of-head false detections."""
        if face_roi_gray is None or getattr(face_roi_gray, "size", 0) == 0 or w <= 0 or h <= 0 or int(h * 0.6) <= 0:
            self.head_yaw_fallback_count += 1
            self.get_logger().error(
                f"_estimate_head_yaw: invalid ROI (face_roi_gray={face_roi_gray is not None}, "
                f"size={getattr(face_roi_gray, 'size', 0)}, w={w}, h={h})"
            )
            return 0.0, False

        try:
            eye_slice = face_roi_gray[:int(h * 0.6), :]
            if eye_slice is None or getattr(eye_slice, "size", 0) == 0 or eye_slice.shape[0] == 0 or eye_slice.shape[1] == 0:
                self.head_yaw_fallback_count += 1
                self.get_logger().error("_estimate_head_yaw: empty eye slice")
                return 0.0, False

            roi = cv2.resize(eye_slice, (96, 54), interpolation=cv2.INTER_AREA) if w > 96 else eye_slice
            rw = roi.shape[1]
            eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.12, minNeighbors=3, minSize=(10, 10))
            if len(eyes) >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                left_eye_center = eyes_sorted[0][0] + eyes_sorted[0][2] / 2.0
                right_eye_center = eyes_sorted[-1][0] + eyes_sorted[-1][2] / 2.0
                eye_midpoint = (left_eye_center + right_eye_center) / 2.0
                face_center = rw / 2.0
                yaw_deg = float(((eye_midpoint - face_center) / face_center) * 40.0)
                return yaw_deg, True
            elif len(eyes) == 1:
                eye_x = eyes[0][0] + eyes[0][2] / 2.0
                yaw_deg = -25.0 if eye_x < rw / 2.0 else 25.0
                return yaw_deg, True
            # If no eyes are detected, user is NOT looking at the robot!
            return 45.0, False
        except Exception as exc:
            self.detector_exception_count += 1
            self.head_yaw_fallback_count += 1
            self.get_logger().error(f"_estimate_head_yaw exception: {exc}")
            return 0.0, False

    def _detect_facial_emotion(self, face_roi_gray, w, h, yaw: float = 0.0, eyes_found: bool = False) -> str:
        """Determines emotion (happy, surprised, focused, neutral) based on mouth contrast, smile and gaze geometry."""
        if face_roi_gray is None or getattr(face_roi_gray, "size", 0) == 0 or w <= 0 or h <= 0 or int(h * 0.5) <= 0:
            return "neutral"

        try:
            mouth_slice = face_roi_gray[int(h * 0.5):, :]
            if mouth_slice is None or getattr(mouth_slice, "size", 0) == 0 or mouth_slice.shape[0] == 0 or mouth_slice.shape[1] == 0:
                return "neutral"

            lower_face = cv2.resize(mouth_slice, (96, 48), interpolation=cv2.INTER_AREA) if w > 96 else mouth_slice
            smiles = self.smile_cascade.detectMultiScale(lower_face, scaleFactor=1.65, minNeighbors=10, minSize=(15, 15))
            if len(smiles) > 0:
                return "happy"

            # Surprise detection: open oral cavity with dark center contrast + high variance in mouth region
            try:
                mouth_region = lower_face[int(lower_face.shape[0] * 0.3):int(lower_face.shape[0] * 0.9), :]
                if mouth_region.size > 0:
                    mean_val = float(np.mean(mouth_region))
                    std_val = float(np.std(mouth_region))
                    mh, mw = mouth_region.shape[:2]
                    center_patch = mouth_region[int(mh * 0.3):int(mh * 0.7), int(mw * 0.3):int(mw * 0.7)]
                    if center_patch.size > 0:
                        center_mean = float(np.mean(center_patch))
                        if center_mean < (mean_val - 18.0) and std_val > 22.0:
                            return "surprised"
            except Exception:
                pass

            # Focused detection: direct frontal gaze with eyes clearly tracked and low head yaw
            if eyes_found and abs(yaw) <= 8.0:
                return "focused"

            return "neutral"
        except Exception as exc:
            self.detector_exception_count += 1
            self.get_logger().error(f"_detect_facial_emotion exception: {exc}")
            return "neutral"

    def _draw_hud(self, frame):
        """Son tespit sonuçlarını kareye çizer (her karede çağrılır)."""
        if not hasattr(self, '_cached_hud'):
            return
        for hud in self._cached_hud:
            x, y, w, h = hud["x"], hud["y"], hud["w"], hud["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), hud["color"], 2)
            cv2.putText(frame, hud["text"], (x, max(22, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud["color"], 2)

    def _should_process(self, frame_count: int) -> bool:
        """True on the frames the heavy detection pipeline is allowed to run."""
        return (frame_count % self.process_every_n) == 0

    def _face_detect_kwargs(self) -> dict:
        """Cascade tuning, from the declared parameters rather than hardcoded literals."""
        return {
            "scaleFactor": self.scale_factor,
            "minNeighbors": self.min_neighbors,
            "minSize": (self.min_size, self.min_size),
        }

    def image_callback(self, msg: Image):
        try:
            frame = imgmsg_to_bgr(msg)
            if frame is None:
                return
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1

        # Canlı pencere — her karede göster (son tespit sonuçlarıyla)
        if self.show_debug:
            display = frame.copy()
            self._draw_hud(display)
            cv2.imshow("ASTRO Vision Debug", display)
            cv2.waitKey(1)

        # Ağır işleme yalnızca process_every_n karede bir (varsayılan: her kare)
        if not self._should_process(self._frame_count):
            return

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Scale to 640px for high detection accuracy on Jetson (preserving facial landmarks)
        scale_ratio = 640.0 / float(frame_w) if frame_w > 640 else 1.0
        
        # YuNet is trained on colour, so detection runs on the resized BGR frame; the
        # cascade fallback converts to grey itself.
        if scale_ratio < 1.0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)
        else:
            small_frame = frame

        detected_faces = self.face_detector.detect(small_frame)
        self.real_detections_count += len(detected_faces)

        # Map bounding boxes back to original resolution (the confidence is scale-free)
        if len(detected_faces) > 0 and scale_ratio < 1.0:
            faces = [[int(x / scale_ratio), int(y / scale_ratio), int(w / scale_ratio), int(h / scale_ratio), conf] for (x, y, w, h, conf) in detected_faces]
        else:
            faces = [list(f) for f in detected_faces]

        # Sort detected faces by bounding box area (largest face first)
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

        face_list = []
        is_looking = False
        user_distance = 0.0
        head_yaw = 0.0
        face_camera_azimuth = 0.0
        detected_emotion = "neutral"
        top_recognized_person = {"name": "Misafir", "title": "Ziyaretçi", "formal_title": "Misafir", "confidence": 0.0, "is_known": False}
        hud_cache = []

        # Telemetry: 5-second interval log
        now = time.time()
        if now - self._last_perf_log_time >= 5.0:
            self.get_logger().info(
                f"[VISION PERF] invalid_roi={self.invalid_roi_count}, "
                f"clipped_bbox={self.clipped_bbox_count}, "
                f"yaw_fallback={self.head_yaw_fallback_count}, "
                f"exceptions={self.detector_exception_count}, "
                f"real_detections={self.real_detections_count}, "
                f"synthetic_hold_detections={self.synthetic_hold_detections_count}"
            )
            self._last_perf_log_time = now

        for idx, (x, y, w, h, detection_conf) in enumerate(faces):
            try:
                orig_x, orig_y, orig_w, orig_h = x, y, w, h
                x = max(0, min(x, frame_w - 1))
                y = max(0, min(y, frame_h - 1))
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)

                if (x, y, w, h) != (orig_x, orig_y, orig_w, orig_h):
                    self.clipped_bbox_count += 1

                if w <= 0 or h <= 0:
                    self.invalid_roi_count += 1
                    continue

                face_roi_gray = gray[y:y + h, x:x + w]
                face_roi_bgr = frame[y:y + h, x:x + w]

                if (
                    face_roi_gray is None
                    or getattr(face_roi_gray, "size", 0) == 0
                    or face_roi_gray.shape[0] == 0
                    or face_roi_gray.shape[1] == 0
                ):
                    self.invalid_roi_count += 1
                    continue

                # 1. 3D Head Yaw & Eye Verification
                yaw, eyes_found = self._estimate_head_yaw(face_roi_gray, w, h)
                head_yaw = yaw

                # Camera Optical Axis Azimuth Angle (HFOV ~ 72°, half = 36°)
                # In ROS body frame: Image Right (+X) is Robot Right (-Yaw), Image Left (-X) is Robot Left (+Yaw)
                face_center_x = x + (w / 2.0)
                norm_offset = (face_center_x - (frame_w / 2.0)) / (frame_w / 2.0)
                cam_azimuth = float(-norm_offset * 36.0)
                if idx == 0:
                    face_camera_azimuth = cam_azimuth

                # 2. 3D Distance
                dist_m = self._estimate_distance(x, y, w, h, frame_w, frame_h)
                user_distance = dist_m

                # 3. Direct Gaze: Eyes MUST be visible AND yaw <= 22 degrees AND strictly in Social Zone (0.40m - 2.20m)
                direct_gaze = eyes_found and (abs(yaw) <= 22.0) and (0.40 <= dist_m <= 2.20)
                if direct_gaze:
                    is_looking = True

                # 4. Face Recognition Matching
                recog_name, recog_conf, recog_meta = self.face_recognizer.recognize_face(face_roi_bgr)
                is_known = (recog_name is not None and recog_conf >= 0.45)
                if is_known and recog_conf > top_recognized_person["confidence"]:
                    top_recognized_person = {
                        "name": recog_name,
                        "title": recog_meta.get("title", "Tanınan Kişi"),
                        "formal_title": recog_meta.get("formal_title", recog_name),
                        "confidence": recog_conf,
                        "is_known": True,
                        "distance_m": round(dist_m, 2)
                    }

                # 5. Emotion Detection
                detected_emotion = self._detect_facial_emotion(face_roi_gray, w, h, yaw=yaw, eyes_found=eyes_found)

                face_list.append({
                    "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                    "confidence": round(float(detection_conf), 2),
                    "frame_width": int(frame_w), "frame_height": int(frame_h),
                    "yaw_deg": round(yaw, 1),
                    "camera_azimuth_deg": round(cam_azimuth, 1),
                    "distance_m": round(dist_m, 2),
                    "looking_at_robot": direct_gaze,
                    "emotion": detected_emotion,
                    "recognized_name": recog_name if is_known else None,
                    "recognized_title": recog_meta.get("formal_title") if is_known else None
                })

                # Build HUD overlay data
                color_map = {
                    "happy": (0, 255, 0),
                    "surprised": (255, 255, 0),
                    "neutral": (0, 200, 255),
                    "sad": (0, 0, 255)
                }
                box_color = (0, 215, 255) if is_known else color_map.get(detected_emotion, (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                tag_name = f"★ {recog_name} ({recog_meta.get('formal_title', '')})" if is_known else detected_emotion.upper()
                gaze_txt = "BANA BAKIYOR" if direct_gaze else (f"YANA ({yaw:.0f}°)" if eyes_found else "BAKMIYOR (GÖZ YOK)")
                hud_text = f"{tag_name} | {gaze_txt} | {dist_m:.2f}m"
                cv2.putText(frame, hud_text, (x, max(22, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                # Cache for non-processed frames
                hud_cache.append({"x": x, "y": y, "w": w, "h": h, "color": box_color, "text": hud_text})
            except Exception as exc:
                self.detector_exception_count += 1
                self.get_logger().error(f"Detector exception processing face: {exc}")
                continue

        # Son tespit sonuçlarını cache'le (aradaki karelerde çizilecek)
        self._cached_hud = hud_cache

        self._gaze_history.append(is_looking)
        smoothed_looking = (self._gaze_history.count(True) >= 2)

        self._emotion_history.append(detected_emotion)
        # Dominant emotion
        smoothed_emotion = max(set(self._emotion_history), key=self._emotion_history.count)

        # Publishers
        faces_msg = String()
        faces_msg.data = json.dumps(face_list)
        self.pub_faces.publish(faces_msg)

        person_msg = Bool()
        person_msg.data = len(faces) > 0
        self.pub_person.publish(person_msg)

        looking_msg = Bool()
        looking_msg.data = smoothed_looking
        self.pub_looking.publish(looking_msg)

        yaw_msg = Float32()
        yaw_msg.data = float(face_camera_azimuth)
        self.pub_yaw.publish(yaw_msg)

        dist_msg = Float32()
        dist_msg.data = float(user_distance)
        self.pub_distance.publish(dist_msg)

        emotion_msg = String()
        emotion_msg.data = smoothed_emotion
        self.pub_emotion.publish(emotion_msg)

        recog_msg = String()
        recog_msg.data = json.dumps(top_recognized_person)
        self.pub_recognized_person.publish(recog_msg)

        # Diagnostic logger on recognized person
        if not hasattr(self, '_last_recog_logged_name'):
            self._last_recog_logged_name = None
            self._last_recog_logged_time = 0.0

        now_t = self.get_clock().now().nanoseconds / 1e9
        if top_recognized_person["is_known"]:
            recog_p_name = top_recognized_person["name"]
            if (recog_p_name != self._last_recog_logged_name) or (now_t - self._last_recog_logged_time > 10.0):
                self._last_recog_logged_name = recog_p_name
                self._last_recog_logged_time = now_t
                self.get_logger().info(f"👤 [Yüz Tanındı]: {recog_p_name} ({top_recognized_person['formal_title']}) — Güven: %{int(top_recognized_person['confidence']*100)}, Mesafe: {user_distance:.2f}m")

        # Diagnostic logger on gaze state change
        if not hasattr(self, '_prev_looking_log'):
            self._prev_looking_log = False
        if is_looking != self._prev_looking_log:
            self._prev_looking_log = is_looking
            if is_looking and (0.35 <= user_distance <= 2.50):
                known_tag = f" — [{top_recognized_person['formal_title']}]" if top_recognized_person["is_known"] else ""
                self.get_logger().info(f"👀 [Göz Teması]: Kullanıcı algılandı! (Mesafe: {user_distance:.2f}m, Açı: {head_yaw:.1f}°){known_tag}")


        # Publish Images
        try:
            face_img_msg = bgr_to_imgmsg(frame, msg.header)
            self.pub_image.publish(face_img_msg)
        except Exception as _exc:
            self.get_logger().debug(f"image_callback: yok sayılan hata ({_exc})")


def main(args=None):
    rclpy.init(args=args)
    node = SpatialVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt as _exc:
        _LOG.debug("main: yok sayılan hata (%s)", _exc)
    finally:
        if node.show_debug:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
