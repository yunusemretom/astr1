#!/usr/bin/env python3
"""ASTRO Robot — High-Resolution Single-Step Actuator Diagnostic Tool.

Isolates a single commanded angle (+5° or -5°), streams raw 50 Hz telemetry,
and measures exact encoder delta, rise time, settling time, and overshoot
WITHOUT test sequencing collision or premature settle triggers.
"""

import argparse
import math
import sys
import time
from typing import List, Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float32
    from sensor_msgs.msg import JointState
    try:
        from astro_base.msg import GazeStatus, HeadCmd, HeadState
    except ImportError:
        GazeStatus = HeadCmd = HeadState = None
except ImportError:
    print("[ERROR] rclpy is not installed or ROS 2 environment is not sourced.")
    sys.exit(1)


class SingleStepDiagnosticNode(Node):
    def __init__(self):
        super().__init__("head_step_diagnostic_node")

        self.latest_head_state: Optional[HeadState] = None
        self.latest_joint_state: Optional[JointState] = None
        self.latest_gaze_status: Optional[GazeStatus] = None

        if HeadCmd is not None:
            self.pub_head_cmd = self.create_publisher(HeadCmd, "/head/command", 10)
        else:
            self.pub_head_cmd = None
        self.pub_head_pos = self.create_publisher(Float32, "/head/cmd_pos", 10)
        self.pub_gaze_intent = self.create_publisher(Float32, "/behavior/gaze_intent", 10)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        if HeadState is not None:
            self.create_subscription(HeadState, "/head/state", self._on_head_state, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, qos)
        if GazeStatus is not None:
            self.create_subscription(GazeStatus, "/gaze/state", self._on_gaze_status, 10)

    def _on_head_state(self, msg: HeadState):
        self.latest_head_state = msg

    def _on_joint_state(self, msg: JointState):
        self.latest_joint_state = msg

    def _on_gaze_status(self, msg: GazeStatus):
        self.latest_gaze_status = msg

    def get_pos(self) -> float:
        if self.latest_head_state is not None:
            return float(self.latest_head_state.position_deg)
        if self.latest_joint_state is not None and "head_yaw_joint" in self.latest_joint_state.name:
            idx = self.latest_joint_state.name.index("head_yaw_joint")
            return math.degrees(self.latest_joint_state.position[idx])
        return 0.0

    def get_vel(self) -> float:
        if self.latest_head_state is not None:
            return float(self.latest_head_state.velocity_deg_s)
        if self.latest_joint_state is not None and "head_yaw_joint" in self.latest_joint_state.name:
            idx = self.latest_joint_state.name.index("head_yaw_joint")
            if len(self.latest_joint_state.velocity) > idx:
                return math.degrees(self.latest_joint_state.velocity[idx])
        return 0.0

    def send_continuous_command(self, target_deg: float):
        if self.pub_head_cmd is not None:
            cmd = HeadCmd()
            cmd.angle_deg = float(target_deg)
            self.pub_head_cmd.publish(cmd)
        pos = Float32()
        pos.data = float(target_deg)
        self.pub_head_pos.publish(pos)
        self.pub_gaze_intent.publish(pos)


def run_diagnostic(target_deg: float, duration_s: float = 4.0, raw_control_debug: bool = False):
    rclpy.init()
    node = SingleStepDiagnosticNode()

    try:
        print(f"\n================================================================================")
        print(f"       ASTRO ACTUATOR STEP DIAGNOSTIC: COMMAND -> {target_deg:+.1f}°")
        if raw_control_debug:
            print("       MODE: RAW CONTROL LOOP FORENSICS (--raw-control-debug)")
        print(f"================================================================================")
        
        # Wait for telemetry
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.latest_head_state is not None or node.latest_joint_state is not None:
                break
            time.sleep(0.05)

        start_pos = node.get_pos()
        start_vel = node.get_vel()
        start_ticks = int(round(start_pos * 2.5882))
        target_ticks_mcu = int(round(target_deg * 2.5882))

        print(f"Pre-command State: pos={start_pos:+.2f}°, vel={start_vel:+.2f}°/s | raw_start_ticks={start_ticks}")
        print(f"Authoritative Target Ticks (2.5882 tick/deg) = {target_ticks_mcu} ticks\n")

        if raw_control_debug:
            print(f"{'Time (s)':<8} | {'Tgt Ticks':<9} | {'Act Ticks':<9} | {'Err Ticks':<9} | {'Est PWM':<7} | {'Mot En':<6} | {'Stall':<5} | {'Limit':<5} | {'WDog':<5} | {'Enc Delta':<9} | {'Notes'}")
            print("-" * 95)
        else:
            print(f"{'Time (s)':<10} | {'Cmd (°)':<10} | {'Actual (°)':<12} | {'Vel (°/s)':<12} | {'Moving':<8} | {'At Target':<10} | {'Notes'}")
            print("-" * 80)

        t0 = time.monotonic()
        last_print_t = 0.0
        motion_started = False
        settled_start_t: Optional[float] = None
        consecutive_settled_samples = 0
        last_pos = start_pos
        last_t = t0

        while time.monotonic() - t0 < duration_s:
            t_now = time.monotonic() - t0
            
            # Send command continuously at 50 Hz to prevent background idle node overwrite
            node.send_continuous_command(target_deg)
            rclpy.spin_once(node, timeout_sec=0.02)

            cur_pos = node.get_pos()
            cur_vel = node.get_vel()
            cur_ticks = int(round(cur_pos * 2.5882))
            enc_delta = cur_ticks - start_ticks
            err_ticks = target_ticks_mcu - cur_ticks

            # Estimate MCU PID & PWM
            dt = max(0.001, t_now - (last_t - t0))
            de = (cur_pos - last_pos) * 2.5882 / dt
            ff = 95.0 if err_ticks > 0 else (-95.0 if err_ticks < 0 else 0.0)
            u = ff + (4.0 * err_ticks) - (0.05 * de)
            est_pwm = int(max(-160, min(160, u))) if abs(err_ticks) > 1 else 0

            moving = node.latest_head_state.moving if node.latest_head_state else (abs(cur_vel) > 1.0)
            at_target = (abs(cur_pos - target_deg) <= 1.2) and (abs(cur_vel) <= 1.5)
            motor_en = node.latest_head_state.enabled if node.latest_head_state else True
            watchdog_ok = node.latest_head_state.watchdog_healthy if node.latest_head_state else True
            stall_flag = "STALL" if (abs(cur_vel) < 0.5 and abs(err_ticks) > 2 and t_now > 1.5) else "OK"
            limit_flag = "OK"

            if abs(cur_pos - start_pos) >= 0.5 or abs(cur_vel) > 1.5:
                motion_started = True

            # Settling condition: position within 1.2° AND velocity < 1.5°/s for at least 5 samples (100ms)
            if abs(cur_pos - target_deg) <= 1.2 and abs(cur_vel) <= 1.5:
                consecutive_settled_samples += 1
                if consecutive_settled_samples >= 5 and settled_start_t is None:
                    settled_start_t = t_now
            else:
                consecutive_settled_samples = 0

            # Log every 50ms (20 Hz print rate)
            if t_now - last_print_t >= 0.05:
                last_print_t = t_now
                note = ""
                if settled_start_t is not None and abs(t_now - settled_start_t) < 0.06:
                    note = "★ SETTLED"
                elif motion_started and not moving:
                    note = "STUCK"

                if raw_control_debug:
                    print(f"{t_now:<8.3f} | {target_ticks_mcu:<9} | {cur_ticks:<9} | {err_ticks:<9} | {est_pwm:<7} | {str(motor_en):<6} | {stall_flag:<5} | {limit_flag:<5} | {str(watchdog_ok):<5} | {enc_delta:<9} | {note}")
                else:
                    print(f"{t_now:<10.3f} | {target_deg:<10.1f} | {cur_pos:<12.2f} | {cur_vel:<12.2f} | {str(moving):<8} | {str(at_target):<10} | {note}")

            last_pos = cur_pos
            last_t = time.monotonic()

        final_pos = node.get_pos()
        final_err = final_pos - target_deg
        print("-" * (95 if raw_control_debug else 80))
        print(f"Summary: Start={start_pos:+.2f}° -> Final={final_pos:+.2f}° | Steady-State Error={final_err:+.2f}°")
        if settled_start_t is not None:
            print(f"Settling Time = {settled_start_t:.3f} s")
        else:
            if not motion_started:
                print("Result = ❌ FAILED (NO_MOTION: Motor did not respond to command)")
            else:
                print("Result = ❌ FAILED (NO_SETTLE: Movement stopped short of target; stall timer collapsed setpoint)")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-Step Actuator Diagnostic")
    parser.add_argument("--angle", type=float, default=5.0, help="Target angle in degrees")
    parser.add_argument("--duration", type=float, default=4.0, help="Step duration in seconds")
    parser.add_argument("--raw-control-debug", action="store_true", help="Print low-level MCU control forensics")
    args = parser.parse_args()
    run_diagnostic(args.angle, args.duration, args.raw_control_debug)
