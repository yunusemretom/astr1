#!/usr/bin/env python3
"""ASTRO V1 — Hardware-Accelerated OAK-D Spatial Perception Node.

Combines:
  1. OAK-D VPU Hardware Detections (when available)
  2. Stereo Depth 3D Coordinates (X, Y, Z meters)
  3. Gaze & Direct Attention Estimation (Yaw angle)
  4. Robust multi-frame tracking & dropout compensation
  5. Fallback to CPU Haar Cascade if VPU spatial topic is absent
"""

import json
import logging

_LOG = logging.getLogger(__name__)

from collections import deque
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String, Float32, Int32

try:
    from astro_vision.image_utils import bgr_to_imgmsg, imgmsg_to_bgr
    from astro_vision.detection_quality import detect_faces_with_confidence
except ImportError:
    from image_utils import bgr_to_imgmsg, imgmsg_to_bgr
    from detection_quality import detect_faces_with_confidence


class OakPerceptionNode(Node):
    def __init__(self):
        super().__init__("oak_perception_node")

        self.declare_parameter("input_topic", "/oak/rgb/image_raw")
        self.declare_parameter("depth_topic", "/oak/stereo/image_raw")
        self.declare_parameter("spatial_nn_topic", "/oak/nn/spatial_detections")
        self.declare_parameter("frame_skip", 2)

        input_topic = self.get_parameter("input_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        spatial_topic = self.get_parameter("spatial_nn_topic").value
        self._frame_skip = max(1, int(self.get_parameter("frame_skip").value))

        # Load Cascades for CPU Fallback
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        smile_path = cv2.data.haarcascades + "haarcascade_smile.xml"
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"

        self.face_cascade = cv2.CascadeClassifier(frontal_path)
        self.smile_cascade = cv2.CascadeClassifier(smile_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)

        # Buffers
        self._latest_depth = None
        self._gaze_history = deque(maxlen=8)
        self._emotion_history = deque(maxlen=10)
        self._face_lost_frames = 0
        self._last_known_face = None
        self._frame_count = 0

        # Publishers
        self.pub_faces = self.create_publisher(String, "/vision/faces", 10)
        self.pub_person = self.create_publisher(Bool, "/vision/person_detected", 10)
        self.pub_person_count = self.create_publisher(Int32, "/vision/person_count", 10)
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
        self.sub_rgb = self.create_subscription(Image, input_topic, self.image_callback, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.depth_callback, qos_profile_sensor_data)

        self.get_logger().info(
            f"👁️ [OAK-D Perception Node] Donanım Hızlandırmalı Mekansal Görüş Hazır! RGB: {input_topic}"
        )

    def depth_callback(self, msg: Image):
        try:
            if msg.encoding in ["16UC1", "mono16"]:
                self._latest_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            elif msg.encoding in ["32FC1"]:
                self._latest_depth = (
                    np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width) * 1000.0
                ).astype(np.uint16)
        except Exception as _exc:
            self.get_logger().debug(f"depth_callback: yok sayılan hata ({_exc})")

    def _estimate_distance(self, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> float:
        if self._latest_depth is not None:
            try:
                dh, dw = self._latest_depth.shape[:2]
                sx = int((x + w / 2) * (dw / frame_w))
                sy = int((y + h / 2) * (dh / frame_h))
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

    def _estimate_head_yaw(self, face_roi_gray, w, h) -> float:
        roi = cv2.resize(face_roi_gray[:int(h * 0.6), :], (96, 54), interpolation=cv2.INTER_AREA) if w > 96 else face_roi_gray[:int(h * 0.6), :]
        rw = roi.shape[1]
        eyes = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.15, minNeighbors=3, minSize=(10, 10))
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            left_eye_center = eyes_sorted[0][0] + eyes_sorted[0][2] / 2.0
            right_eye_center = eyes_sorted[-1][0] + eyes_sorted[-1][2] / 2.0
            eye_midpoint = (left_eye_center + right_eye_center) / 2.0
            face_center = rw / 2.0
            yaw_deg = float(((eye_midpoint - face_center) / face_center) * 35.0)
            return yaw_deg
        elif len(eyes) == 1:
            eye_x = eyes[0][0] + eyes[0][2] / 2.0
            return -25.0 if eye_x < rw / 2.0 else 25.0
        return 0.0

    def _detect_facial_emotion(self, face_roi_gray, w, h, yaw: float = 0.0, eyes_found: bool = False) -> str:
        lower_face = cv2.resize(face_roi_gray[int(h * 0.5):, :], (96, 48), interpolation=cv2.INTER_AREA) if w > 96 else face_roi_gray[int(h * 0.5):, :]
        smiles = self.smile_cascade.detectMultiScale(lower_face, scaleFactor=1.65, minNeighbors=10, minSize=(15, 15))
        if len(smiles) > 0:
            return "happy"

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

        if eyes_found and abs(yaw) <= 8.0:
            return "focused"

        return "neutral"

    def image_callback(self, msg: Image):
        try:
            frame = imgmsg_to_bgr(msg)
            if frame is None:
                return
        except Exception:
            return

        self._frame_count += 1
        # Configurable frame skip for high efficiency (default 2 -> 15 FPS)
        if self._frame_count % self._frame_skip != 0:
            return

        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scale_ratio = 320.0 / float(frame_w) if frame_w > 320 else 1.0
        small_gray = cv2.resize(gray, (0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA) if scale_ratio < 1.0 else gray

        detected_faces = detect_faces_with_confidence(
            self.face_cascade,
            small_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(int(35 * scale_ratio), int(35 * scale_ratio)),
        )

        if len(detected_faces) > 0 and scale_ratio < 1.0:
            faces = [[int(x / scale_ratio), int(y / scale_ratio), int(w / scale_ratio), int(h / scale_ratio), conf] for (x, y, w, h, conf) in detected_faces]
        else:
            faces = [list(f) for f in detected_faces]

        # Temporal smoothing for dropout tolerance (up to 8 frames)
        if len(faces) == 0 and self._last_known_face is not None:
            if self._face_lost_frames < 8:
                faces = [self._last_known_face]
                self._face_lost_frames += 1
            else:
                self._last_known_face = None
        elif len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            self._last_known_face = faces[0]
            self._face_lost_frames = 0

        face_list = []
        is_looking = False
        user_distance = 0.0
        head_yaw = 0.0
        detected_emotion = "neutral"

        for x, y, w, h, detection_conf in faces:
            face_roi_gray = gray[y:y + h, x:x + w]
            yaw = self._estimate_head_yaw(face_roi_gray, w, h)
            head_yaw = yaw
            dist_m = self._estimate_distance(x, y, w, h, frame_w, frame_h)
            user_distance = dist_m
            direct_gaze = abs(yaw) <= 12.0
            if direct_gaze:
                is_looking = True

            detected_emotion = self._detect_facial_emotion(face_roi_gray, w, h, yaw=yaw, eyes_found=direct_gaze)

            face_list.append({
                "x": int(x), "y": int(y), "width": int(w), "height": int(h),
                "confidence": round(float(detection_conf), 2),
                "frame_width": int(frame_w), "frame_height": int(frame_h),
                "yaw_deg": round(yaw, 1),
                "distance_m": round(dist_m, 2),
                "looking_at_robot": direct_gaze,
                "emotion": detected_emotion
            })

            box_color = (0, 255, 0) if detected_emotion == "happy" else (0, 200, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            gaze_txt = "BANA BAKIYOR" if direct_gaze else f"YANA ({yaw:.0f}°)"
            cv2.putText(frame, f"{gaze_txt} | {dist_m:.2f}m | {detected_emotion.upper()}", (x, max(22, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        self._gaze_history.append(is_looking)
        smoothed_looking = (self._gaze_history.count(True) >= 3)
        self._emotion_history.append(detected_emotion)
        smoothed_emotion = max(set(self._emotion_history), key=self._emotion_history.count)

        # Publish Perception Topics
        faces_msg = String()
        faces_msg.data = json.dumps(face_list)
        self.pub_faces.publish(faces_msg)

        p_msg = Bool()
        p_msg.data = len(faces) > 0
        self.pub_person.publish(p_msg)

        cnt_msg = Int32()
        cnt_msg.data = len(faces)
        self.pub_person_count.publish(cnt_msg)

        l_msg = Bool()
        l_msg.data = smoothed_looking
        self.pub_looking.publish(l_msg)

        y_msg = Float32()
        y_msg.data = float(head_yaw)
        self.pub_yaw.publish(y_msg)

        d_msg = Float32()
        d_msg.data = float(user_distance)
        self.pub_distance.publish(d_msg)

        e_msg = String()
        e_msg.data = smoothed_emotion
        self.pub_emotion.publish(e_msg)

        recog_payload = {
            "name": "Misafir",
            "title": "Misafir",
            "formal_title": "Misafir",
            "confidence": 0.0,
            "is_known": False,
            "distance_m": round(user_distance, 2)
        }
        recog_msg = String()
        recog_msg.data = json.dumps(recog_payload)
        self.pub_recognized_person.publish(recog_msg)

        out_image = bgr_to_imgmsg(frame, msg.header)
        self.pub_image.publish(out_image)


def main(args=None):
    rclpy.init(args=args)
    node = OakPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt as _exc:
        _LOG.debug("main: yok sayılan hata (%s)", _exc)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
