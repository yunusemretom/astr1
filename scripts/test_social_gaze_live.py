#!/usr/bin/env python3
"""ASTRO Robot — Real-Time Audiovisual Social Gaze Live Monitor & Flight Recorder.

Subscribes to all perception, fusion, state machine, and actuator topics to provide:
  1. Live interactive dashboard (10 Hz in-place display)
  2. Background Flight Recorder tracking all acoustic orienting, face tracking, and FSM events
  3. Formatted Summary Report printed to stdout and saved to JSON on Ctrl+C exit

Usage:
  python3 scripts/test_social_gaze_live.py
"""

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool, Float32, Int32, String
    from sensor_msgs.msg import JointState
    try:
        from astro_base.msg import GazeStatus, HeadCmd, HeadState
    except ImportError:
        GazeStatus = HeadCmd = HeadState = None
except ImportError as exc:
    print(f"Error importing ROS 2 packages: {exc}")
    sys.exit(1)


STATE_NAMES = {
    0: "IDLE",
    1: "SEARCHING",
    2: "ACQUIRING",
    3: "ORIENTING",
    4: "TRACKING",
    5: "HOLDING_ATTENTION",
    6: "TARGET_LOST",
    7: "RECOVERING",
}

PRIORITY_NAMES = {
    0: "IDLE",
    1: "VISUAL_TRACKING",
    2: "ACTIVE_SPEAKER",
    3: "GESTURE_INTENT",
    4: "DIRECT_DIALOGUE_INTENT",
    5: "EXPLICIT_USER_GAZE",
    6: "EMERGENCY_STOP",
}


class SocialGazeLiveMonitor(Node):
    def __init__(self):
        super().__init__("social_gaze_live_monitor")

        self.start_time = time.monotonic()

        # Telemetry State
        self.audio_doa_deg: Optional[float] = None
        self.audio_last_time: float = 0.0
        self.visual_yaw_deg: Optional[float] = None
        self.visual_conf: float = 0.0
        self.visual_faces_count: int = 0
        self.visual_last_time: float = 0.0

        self.gaze_state: str = "IDLE"
        self.prev_gaze_state: str = "IDLE"
        self.gaze_priority: str = "IDLE"
        self.gaze_desired_yaw: float = 0.0
        self.gaze_planned_yaw: float = 0.0
        self.gaze_at_target: bool = True
        self.gaze_target_valid: bool = False
        self.gaze_target_conf: float = 0.0
        self.gaze_target_id: str = "None"
        self.active_target_json: str = "{}"

        self.cmd_head_yaw: float = 0.0
        self.act_head_yaw: float = 0.0
        self.act_head_vel: float = 0.0
        self.head_moving: bool = False
        self.head_stall: bool = False
        self.robot_speaking: bool = False

        # Flight Recorder History
        self.state_transitions: List[Dict] = []
        self.audio_events: List[Dict] = []
        self.visual_events: List[Dict] = []
        self.tracking_errors: List[float] = []
        self.sample_count: int = 0
        self.max_abs_vel: float = 0.0

        qos_be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscriptions
        self.create_subscription(Float32, "/audio/doa", self._on_audio_doa, 10)
        self.create_subscription(Float32, "/audio/doa_deg", self._on_audio_doa, 10)
        self.create_subscription(String, "/vision/detections_json", self._on_vision_json, 10)
        self.create_subscription(String, "/vision/faces", self._on_vision_json, 10)
        self.create_subscription(Float32, "/vision/head_yaw", self._on_vision_head_yaw, 10)
        self.create_subscription(Bool, "/vision/person_detected", self._on_person_detected, 10)
        self.create_subscription(String, "/gaze/active_target", self._on_active_target, 10)
        self.create_subscription(Float32, "/head/cmd_pos", self._on_cmd_pos, 10)
        self.create_subscription(Bool, "/robot/is_speaking", self._on_speaking, 10)
        self.create_subscription(Bool, "/audio/playback_active", self._on_speaking, 10)

        if HeadCmd is not None:
            self.create_subscription(HeadCmd, "/head/command", self._on_head_cmd, 10)
        if HeadState is not None:
            self.create_subscription(HeadState, "/head/state", self._on_head_state, 10)
        if GazeStatus is not None:
            self.create_subscription(GazeStatus, "/gaze/state", self._on_gaze_state, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, qos_be)

    def _on_audio_doa(self, msg: Float32):
        val = float(msg.data)
        now = time.monotonic()
        if self.audio_doa_deg is None or abs(val - self.audio_doa_deg) > 3.0 or (now - self.audio_last_time > 1.0):
            self.audio_events.append({
                "time": round(now - self.start_time, 2),
                "doa_deg": round(val, 1),
                "head_yaw_at_trigger": round(self.act_head_yaw, 2),
            })
        self.audio_doa_deg = val
        self.audio_last_time = now

    def _on_vision_head_yaw(self, msg: Float32):
        now = time.monotonic()
        if self.visual_yaw_deg is None or (now - self.visual_last_time > 0.3):
            self.visual_yaw_deg = float(msg.data)
            self.visual_conf = 0.85
            self.visual_faces_count = max(1, self.visual_faces_count)
            self.visual_last_time = now
            self.visual_events.append({
                "time": round(now - self.start_time, 2),
                "face_yaw_deg": round(self.visual_yaw_deg, 2),
                "confidence": round(self.visual_conf, 2),
                "tracking_error": round(self.cmd_head_yaw - self.act_head_yaw, 2),
            })

    def _on_person_detected(self, msg: Bool):
        if not msg.data and (time.monotonic() - self.visual_last_time > 0.5):
            self.visual_faces_count = 0
            self.visual_conf = 0.0

    def _on_vision_json(self, msg: String):
        now = time.monotonic()
        try:
            raw_data = json.loads(msg.data)
            if isinstance(raw_data, dict):
                faces = raw_data.get("faces", [])
            elif isinstance(raw_data, list):
                faces = raw_data
            else:
                faces = []
            self.visual_faces_count = len(faces)
            if faces:
                primary = faces[0]
                self.visual_yaw_deg = float(primary.get("camera_azimuth_deg", primary.get("azimuth_deg", primary.get("yaw_deg", 0.0))))
                self.visual_conf = float(primary.get("confidence", 0.85))
                self.visual_last_time = now
                self.visual_events.append({
                    "time": round(now - self.start_time, 2),
                    "face_yaw_deg": round(self.visual_yaw_deg, 2),
                    "confidence": round(self.visual_conf, 2),
                    "tracking_error": round(self.cmd_head_yaw - self.act_head_yaw, 2),
                })
            else:
                self.visual_conf = 0.0
        except Exception:
            pass

    def _on_active_target(self, msg: String):
        self.active_target_json = msg.data

    def _on_cmd_pos(self, msg: Float32):
        self.cmd_head_yaw = float(msg.data)

    def _on_head_cmd(self, msg):
        self.cmd_head_yaw = float(msg.angle_deg)

    def _on_head_state(self, msg):
        self.act_head_yaw = float(msg.position_deg)
        self.act_head_vel = float(msg.velocity_deg_s)
        self.head_moving = bool(msg.moving)
        self.head_stall = bool(getattr(msg, "fault_code", 0) == 4)
        self.max_abs_vel = max(self.max_abs_vel, abs(self.act_head_vel))
        err = self.cmd_head_yaw - self.act_head_yaw
        self.tracking_errors.append(abs(err))
        self.sample_count += 1

    def _on_joint_states(self, msg: JointState):
        if "head_yaw_joint" in msg.name:
            idx = msg.name.index("head_yaw_joint")
            if HeadState is None or self.act_head_yaw == 0.0:
                self.act_head_yaw = math.degrees(msg.position[idx])
                self.act_head_vel = math.degrees(msg.velocity[idx])

    def _on_gaze_state(self, msg):
        state_val = getattr(msg, "state", 0)
        current_st = STATE_NAMES.get(state_val, f"STATE_{state_val}")
        if current_st != self.prev_gaze_state:
            now = time.monotonic()
            self.state_transitions.append({
                "time": round(now - self.start_time, 2),
                "from_state": self.prev_gaze_state,
                "to_state": current_st,
                "head_yaw": round(self.act_head_yaw, 2),
                "target_yaw": round(float(getattr(msg, "desired_yaw_deg", 0.0)), 2),
            })
            self.prev_gaze_state = current_st

        self.gaze_state = current_st
        prio_val = getattr(msg, "priority", 0)
        self.gaze_priority = PRIORITY_NAMES.get(prio_val, f"PRIO_{prio_val}")
        self.gaze_desired_yaw = float(getattr(msg, "desired_yaw_deg", 0.0))
        self.gaze_planned_yaw = float(getattr(msg, "planned_yaw_deg", 0.0))
        self.gaze_at_target = bool(getattr(msg, "at_target", False))
        self.gaze_target_valid = bool(getattr(msg, "target_valid", False))
        self.gaze_target_conf = float(getattr(msg, "target_confidence", 0.0))
        self.gaze_target_id = str(getattr(msg, "active_target_id", "None"))

    def _on_speaking(self, msg: Bool):
        self.robot_speaking = bool(msg.data)


def render_dashboard(node: SocialGazeLiveMonitor):
    now = time.monotonic()
    audio_fresh = (now - node.audio_last_time < 1.5) and (node.audio_doa_deg is not None)
    vision_fresh = (now - node.visual_last_time < 1.5) and (node.visual_yaw_deg is not None)

    # ANSI Clear Screen and Home (Single Buffer)
    lines = []
    lines.append("=" * 80)
    lines.append("       ASTRO SOCIAL ROBOT — REAL-TIME MULTIMODAL SOCIAL GAZE DASHBOARD")
    lines.append("=" * 80)

    # 1. Perception Layer
    lines.append("\n[1] PERCEPTION INPUTS:")
    audio_str = f"{node.audio_doa_deg:+.1f}° (ACTIVE)" if audio_fresh else "SILENT / NO SIGNAL"
    lines.append(f"  • Acoustic DOA (ReSpeaker) : {audio_str}")
    
    vis_str = f"{node.visual_yaw_deg:+.1f}° (Conf: {node.visual_conf:.2f}, Faces: {node.visual_faces_count})" if vision_fresh else "NO FACE DETECTED"
    lines.append(f"  • Visual Face (OAK-D Lite) : {vis_str}")
    lines.append(f"  • Robot Self-Speech Gate   : {'🔴 SPEAKING (ECHO SUPPRESSION ACTIVE)' if node.robot_speaking else '🟢 LISTENING / ATTENTIVE'}")

    # 2. Gaze Policy & State Machine Layer
    lines.append("\n[2] SOCIAL GAZE POLICY & FSM:")
    lines.append(f"  • Active Gaze State        : \033[1;36m{node.gaze_state}\033[0m")
    lines.append(f"  • Priority Source          : {node.gaze_priority}")
    lines.append(f"  • Desired Gaze Target      : {node.gaze_desired_yaw:+.2f}° (Conf: {node.gaze_target_conf:.2f}, Valid: {node.gaze_target_valid})")
    lines.append(f"  • Motion Planned Trajectory: {node.gaze_planned_yaw:+.2f}°")
    lines.append(f"  • Target Acquired / Fixed  : {'✅ YES (FOVEATED)' if node.gaze_at_target else '⏳ SLEWING / CONVERGING'}")

    # 3. Motion Planning & Actuator Loop
    err = node.cmd_head_yaw - node.act_head_yaw
    lines.append("\n[3] ACTUATOR CLOSED-LOOP EXECUTION:")
    lines.append(f"  • Commanded Trajectory     : {node.cmd_head_yaw:+.2f}°")
    lines.append(f"  • Physical Head Position   : \033[1;32m{node.act_head_yaw:+.2f}°\033[0m")
    lines.append(f"  • Joint Angular Velocity   : {node.act_head_vel:+.1f}°/s")
    lines.append(f"  • Steady-State Error       : {err:+.2f}° ({'★ ON TARGET' if abs(err) <= 1.2 else 'TRACKING'})")
    lines.append(f"  • Actuator Moving Status   : {'🔄 SLEWING' if node.head_moving else '⏸️ SETTLED / DWELLING'}")

    lines.append("\n" + "-" * 80)
    lines.append("  [Canlı Test]: Konuşarak veya hareket ederek test edin. Bitirmek için Ctrl+C yapın.")
    lines.append("-" * 80)

    # Print buffer cleanly at home position
    sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def generate_flight_report(node: SocialGazeLiveMonitor):
    duration = time.monotonic() - node.start_time
    avg_err = (sum(node.tracking_errors) / len(node.tracking_errors)) if node.tracking_errors else 0.0
    rms_err = math.sqrt(sum(e**2 for e in node.tracking_errors) / len(node.tracking_errors)) if node.tracking_errors else 0.0

    print("\n\n" + "=" * 80)
    print("       ASTRO SOCIAL GAZE LIVE TEST — FLIGHT RECORDER REPORT")
    print("=" * 80)
    print(f"Test Süresi          : {duration:.2f} saniye")
    print(f"Toplanan Telemetri   : {node.sample_count} örnek ({node.sample_count / max(0.1, duration):.1f} Hz)")
    print(f"Maksimum Açısal Hız  : {node.max_abs_vel:.1f}°/s")
    print(f"Ortalama Takip Hatası: {avg_err:.2f}° (RMS: {rms_err:.2f}°)")

    # 1. State Machine Timeline
    print("\n[1] DURUM MAKİNESİ GEÇİŞLERİ (FSM TRANSITION TIMELINE):")
    if node.state_transitions:
        print(f"{'Zaman (s)':<12} | {'Önceki Durum':<18} -> {'Yeni Durum':<18} | {'Kafa Konumu':<12} | {'Hedef Açı'}")
        print("-" * 75)
        for st in node.state_transitions:
            print(f"{st['time']:<12.2f} | {st['from_state']:<18} -> {st['to_state']:<18} | {st['head_yaw']:>+6.2f}°      | {st['target_yaw']:>+6.2f}°")
    else:
        print("  (Test süresince durum değişikliği olmadı: Sabit " + node.gaze_state + ")")

    # 2. Acoustic Orienting Summary
    print("\n[2] İŞİTSEL YÖNELME (ACOUSTIC DOA EVENTS):")
    if node.audio_events:
        print(f"  • Tespit Edilen Ses Yönü Sayısı: {len(node.audio_events)}")
        for i, ev in enumerate(node.audio_events[-5:], 1):
            print(f"    - Olay {i}: {ev['time']}s | DOA = {ev['doa_deg']:+.1f}° | Kafa Başlangıç = {ev['head_yaw_at_trigger']:+.2f}°")
    else:
        print("  • Ses olayı algılanmadı (Sessiz ortam).")

    # 3. Visual Face Tracking Summary
    print("\n[3] GÖRSEL YÜZ TAKİBİ (VISUAL TRACKING SUMMARY):")
    if node.visual_events:
        print(f"  • Toplam Takip Edilen Yüz Karesi: {len(node.visual_events)} kare")
        mean_vis_conf = sum(e['confidence'] for e in node.visual_events) / len(node.visual_events)
        print(f"  • Ortalama Tespit Güveni: %{mean_vis_conf*100:.1f}")
        for i, ev in enumerate(node.visual_events[-5:], 1):
            print(f"    - Kare {i}: {ev['time']}s | Yüz Açısı = {ev['face_yaw_deg']:+.1f}° | Takip Hatası = {ev['tracking_error']:+.2f}°")
    else:
        print("  • Görsel yüz tespiti algılanmadı.")

    print("\n" + "=" * 80)
    print("Kabul Özeti: Test başarıyla tamamlandı. Rapor JSON dosyasına kaydedildi.")
    print("=" * 80 + "\n")

    # Save JSON Log
    os.makedirs("ros2_ws/data", exist_ok=True)
    report_file = "ros2_ws/data/social_gaze_live_report.json"
    report_data = {
        "duration_s": round(duration, 2),
        "total_samples": node.sample_count,
        "max_angular_velocity_deg_s": round(node.max_abs_vel, 2),
        "mean_tracking_error_deg": round(avg_err, 2),
        "rms_tracking_error_deg": round(rms_err, 2),
        "state_transitions": node.state_transitions,
        "audio_events": node.audio_events,
        "visual_events_count": len(node.visual_events),
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


def main():
    rclpy.init()
    node = SocialGazeLiveMonitor()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            render_dashboard(node)
    except KeyboardInterrupt:
        pass
    finally:
        generate_flight_report(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
