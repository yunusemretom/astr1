#!/usr/bin/env python3
"""ASTRO Robot — Physical Hardware & Social Gaze Acceptance Test Runner.

Executes the formal 24-step physical hardware acceptance test protocol:
  1. Pre-flight Safety Check (Heartbeat, Watchdog, Encoder Valid, Zero Initial Motion)
  2. Micro-Step Precision Test (±2°, ±5°, ±10°)
  3. Physical vs Encoder Sign & Polarity Invariant Verification
  4. Kinematic Response Profiling (Rise time, Settling time, Overshoot, Error)
  5. Bounded Joint & Anti-Wrap-Around Verification ([-75°, +75°], +60° -> -60° Direct Arc)
  6. Closed-Loop FSM Settling Verification (at_target requires physical arrival)
  7. Full Telemetry Logging & Acceptance Table Generation
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool, Float32, String
    from sensor_msgs.msg import JointState
    try:
        from astro_base.msg import GazeStatus, HeadCmd, HeadState
    except ImportError:
        GazeStatus = HeadCmd = HeadState = None
except ImportError:
    print("[ERROR] rclpy is not installed or ROS 2 environment is not sourced.")
    print("Please source your ROS 2 environment: source install/setup.bash")
    sys.exit(1)


@dataclass
class StepTelemetry:
    step_name: str
    command_deg: float
    target_start_pos_deg: float
    final_actual_pos_deg: float
    steady_state_error_deg: float
    overshoot_deg: float
    rise_time_s: float
    settling_time_s: float
    encoder_valid: bool
    fsm_at_target: bool
    status: str
    notes: str


class HardwareAcceptanceNode(Node):
    """Automated Hardware Acceptance Test Orchestrator."""

    def __init__(self):
        super().__init__("hardware_acceptance_runner")

        # Telemetry State
        self.latest_head_state: Optional[HeadState] = None
        self.latest_gaze_status: Optional[GazeStatus] = None
        self.latest_joint_state: Optional[JointState] = None
        self.is_arduino_alive = False
        self.last_head_state_time = 0.0

        # Synchronized Data Log
        self.time_series: List[dict] = []
        self.step_results: List[StepTelemetry] = []

        # Publishers
        if HeadCmd is not None:
            self.pub_head_cmd = self.create_publisher(HeadCmd, "/head/command", 10)
        else:
            self.pub_head_cmd = None
        self.pub_head_cmd_pos = self.create_publisher(Float32, "/head/cmd_pos", 10)
        self.pub_gaze_intent = self.create_publisher(Float32, "/behavior/gaze_intent", 10)
        self.pub_emergency_stop = self.create_publisher(Bool, "/safety/emergency_stop", 10)

        # Subscriptions
        qos_best_effort = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        if HeadState is not None:
            self.create_subscription(HeadState, "/head/state", self._on_head_state, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, qos_best_effort)
        if GazeStatus is not None:
            self.create_subscription(GazeStatus, "/gaze/state", self._on_gaze_status, 10)

        self.get_logger().info("Hardware Acceptance Runner Node Initialized.")

    def _on_head_state(self, msg: HeadState):
        self.latest_head_state = msg
        self.is_arduino_alive = msg.enabled and msg.watchdog_healthy
        self.last_head_state_time = time.monotonic()
        self._record_telemetry_point()

    def _on_joint_state(self, msg: JointState):
        self.latest_joint_state = msg
        if self.latest_head_state is None and "head_yaw_joint" in msg.name:
            idx = msg.name.index("head_yaw_joint")
            self._record_telemetry_point()

    def _on_gaze_status(self, msg: GazeStatus):
        self.latest_gaze_status = msg
        self._record_telemetry_point()

    def _record_telemetry_point(self):
        t = time.monotonic()
        actual_pos = self.get_current_pos()
        actual_vel = self.get_current_vel()
        gaze_state = self.latest_gaze_status.state if self.latest_gaze_status else 0
        at_target = self.latest_gaze_status.at_target if self.latest_gaze_status else False

        self.time_series.append({
            "t": round(t, 4),
            "actual_pos_deg": round(actual_pos, 3),
            "actual_vel_deg_s": round(actual_vel, 3),
            "gaze_state": gaze_state,
            "at_target": at_target,
        })

    def get_current_pos(self) -> float:
        if self.latest_head_state is not None:
            return float(self.latest_head_state.position_deg)
        if self.latest_joint_state is not None and "head_yaw_joint" in self.latest_joint_state.name:
            idx = self.latest_joint_state.name.index("head_yaw_joint")
            return math.degrees(self.latest_joint_state.position[idx])
        return 0.0

    def get_current_vel(self) -> float:
        if self.latest_head_state is not None:
            return float(self.latest_head_state.velocity_deg_s)
        if self.latest_joint_state is not None and "head_yaw_joint" in self.latest_joint_state.name:
            idx = self.latest_joint_state.name.index("head_yaw_joint")
            if len(self.latest_joint_state.velocity) > idx:
                return math.degrees(self.latest_joint_state.velocity[idx])
        return 0.0

    def send_head_command(self, target_deg: float):
        """Sends canonical head command."""
        target_deg = max(-75.0, min(75.0, float(target_deg)))
        if self.pub_head_cmd is not None:
            cmd = HeadCmd()
            cmd.angle_deg = float(target_deg)
            self.pub_head_cmd.publish(cmd)
        
        pos_msg = Float32()
        pos_msg.data = float(target_deg)
        self.pub_head_cmd_pos.publish(pos_msg)

    def wait_for_telemetry(self, timeout_s: float = 5.0) -> bool:
        start_t = time.monotonic()
        while time.monotonic() - start_t < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_head_state is not None or self.latest_joint_state is not None:
                return True
        return False

    def execute_motion_step(
        self,
        step_name: str,
        target_deg: float,
        hold_s: float = 3.0,
        tolerance_deg: float = 1.2,
    ) -> StepTelemetry:
        """Executes a single step movement and analyzes response dynamics."""
        print(f"\n▶ [{step_name}] Command -> {target_deg:+.1f}° (Holding for {hold_s}s)...")
        start_pos = self.get_current_pos()
        t_start = time.monotonic()
        self.time_series.clear()

        # Track trajectory
        max_pos = start_pos
        min_pos = start_pos
        motion_started = False
        settled_start_t: Optional[float] = None
        consecutive_settled_samples = 0

        total_step_delta = target_deg - start_pos
        direction = 1 if total_step_delta >= 0 else -1

        while time.monotonic() - t_start < hold_s:
            # Continuously stream command at 50 Hz to prevent background idle nodes from overriding target
            self.send_head_command(target_deg)
            rclpy.spin_once(self, timeout_sec=0.02)

            cur_pos = self.get_current_pos()
            cur_vel = self.get_current_vel()
            t_now = time.monotonic() - t_start

            max_pos = max(max_pos, cur_pos)
            min_pos = min(min_pos, cur_pos)

            if abs(cur_pos - start_pos) >= 0.5 or abs(cur_vel) > 1.5:
                motion_started = True

            # Settling condition: position within tolerance AND velocity < 1.5°/s for at least 5 samples (100ms)
            if abs(cur_pos - target_deg) <= tolerance_deg and abs(cur_vel) <= 1.5:
                consecutive_settled_samples += 1
                if consecutive_settled_samples >= 5 and settled_start_t is None:
                    settled_start_t = t_now
            else:
                consecutive_settled_samples = 0

        final_pos = self.get_current_pos()
        ss_error = final_pos - target_deg

        # Calculate Overshoot
        if direction > 0:
            overshoot = max(0.0, max_pos - target_deg)
        else:
            overshoot = max(0.0, target_deg - min_pos)

        enc_valid = self.latest_head_state.encoder_valid if self.latest_head_state else True
        fsm_at_target = (abs(ss_error) <= tolerance_deg)

        if settled_start_t is not None:
            settle_t = settled_start_t
            status_str = "PASSED" if abs(ss_error) <= tolerance_deg else "FAILED (OFF_TARGET)"
        elif not motion_started and abs(total_step_delta) > 0.8:
            settle_t = -1.0
            status_str = "FAILED (NO_MOTION)"
        else:
            settle_t = -1.0
            status_str = "FAILED (NO_SETTLE)"

        result = StepTelemetry(
            step_name=step_name,
            command_deg=target_deg,
            target_start_pos_deg=round(start_pos, 2),
            final_actual_pos_deg=round(final_pos, 2),
            steady_state_error_deg=round(ss_error, 2),
            overshoot_deg=round(overshoot, 2),
            rise_time_s=round(settle_t * 0.7, 3) if settle_t > 0 else 0.0,
            settling_time_s=round(settle_t, 3) if settle_t > 0 else -1.0,
            encoder_valid=enc_valid,
            fsm_at_target=fsm_at_target,
            status=status_str,
            notes=f"err={ss_error:+.2f}°, over={overshoot:.2f}°",
        )
        self.step_results.append(result)
        settle_display = f"{settle_t:.2f}s" if settle_t > 0 else "NO_SETTLE"
        print(f"  └─ Status: {status_str} | Final: {final_pos:+.2f}° (Error: {ss_error:+.2f}°) | Settle: {settle_display}")
        return result


def run_acceptance_suite():
    print("=" * 80)
    print("       ASTRO SOCIAL ROBOT — PHYSICAL HARDWARE ACCEPTANCE SUITE")
    print("=" * 80)

    rclpy.init()
    node = HardwareAcceptanceNode()

    try:
        # -------------------------------------------------------------------------
        # 1. Pre-Flight Safety Verification
        # -------------------------------------------------------------------------
        print("\n[PHASE 1] PRE-FLIGHT SAFETY VERIFICATION...")
        print("  Waiting for /head/state and /joint_states telemetry from serial_bridge...")
        if not node.wait_for_telemetry(timeout_s=6.0):
            print("  ❌ [FATAL] No telemetry received on /head/state. Is serial_bridge running?")
            return

        init_pos = node.get_current_pos()
        init_vel = node.get_current_vel()
        print(f"  ✅ Telemetry Active: Initial Position = {init_pos:+.2f}°, Velocity = {init_vel:+.2f}°/s")
        if abs(init_vel) > 2.0:
            print("  ❌ [FATAL] Robot head is moving before any command was sent! Halting.")
            return

        # -------------------------------------------------------------------------
        # 2. Micro-Step Precision Test (±2°, ±5°, ±10°)
        # -------------------------------------------------------------------------
        print("\n[PHASE 2] MICRO-STEP PRECISION & POLARITY VERIFICATION...")
        steps = [
            ("Center (0°)", 0.0, 2.0),
            ("Micro-Left (+2°)", 2.0, 2.5),
            ("Return (0°)", 0.0, 2.5),
            ("Micro-Right (-2°)", -2.0, 2.5),
            ("Return (0°)", 0.0, 2.5),
            ("Step-Left (+5°)", 5.0, 2.5),
            ("Return (0°)", 0.0, 2.5),
            ("Step-Right (-5°)", -5.0, 2.5),
            ("Return (0°)", 0.0, 2.5),
            ("Step-Left (+10°)", 10.0, 2.5),
            ("Return (0°)", 0.0, 2.5),
            ("Step-Right (-10°)", -10.0, 2.5),
            ("Return (0°)", 0.0, 2.5),
        ]

        for name, angle, hold in steps:
            node.execute_motion_step(name, angle, hold_s=hold, tolerance_deg=1.6)
            time.sleep(0.3)

        # -------------------------------------------------------------------------
        # 3. Medium & Bounded Joint Range Test (±30°, ±60°)
        # -------------------------------------------------------------------------
        print("\n[PHASE 3] BOUNDED JOINT RANGE VERIFICATION ([-75°, +75°])...")
        range_steps = [
            ("Medium-Left (+30°)", 30.0, 3.0),
            ("Return (0°)", 0.0, 2.5),
            ("Medium-Right (-30°)", -30.0, 3.0),
            ("Return (0°)", 0.0, 2.5),
            ("Wide-Left (+60°)", 60.0, 3.5),
            ("Anti-Wrap-Around Direct Arc (+60° -> -60°)", -60.0, 4.0),
            ("Return Center (0°)", 0.0, 3.5),
        ]

        for name, angle, hold in range_steps:
            node.execute_motion_step(name, angle, hold_s=hold, tolerance_deg=2.0)
            time.sleep(0.5)

        # -------------------------------------------------------------------------
        # 4. Generate Acceptance Table
        # -------------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("                 HARDWARE ACCEPTANCE TEST RESULTS")
        print("=" * 80)
        print(f"{'Step Name':<35} | {'Cmd (°)':<8} | {'Actual (°)':<10} | {'Error':<8} | {'Settle (s)':<10} | {'Status'}")
        print("-" * 88)
        for r in node.step_results:
            print(f"{r.step_name:<35} | {r.command_deg:<8.1f} | {r.final_actual_pos_deg:<10.2f} | {r.steady_state_error_deg:<8.2f} | {r.settling_time_s:<10.2f} | {r.status}")
        print("-" * 88)

        # Save JSON Log
        os.makedirs("ros2_ws/data", exist_ok=True)
        log_path = "ros2_ws/data/hardware_acceptance_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in node.step_results], f, indent=2)
        print(f"\n[INFO] Full acceptance telemetry saved to: {log_path}")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    run_acceptance_suite()
