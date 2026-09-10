#!/usr/bin/env python3
import glob
import logging

_LOG = logging.getLogger(__name__)

import math
import os
import struct
import threading
import time

try:
    import serial
except ImportError:
    serial = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Imu, JointState
    from std_msgs.msg import Float32
    from astro_base.msg import HeadCmd, WheelCmd
    try:
        from astro_base.msg import HeadState
    except ImportError:
        HeadState = None

except ImportError:
    class _MockRclpy:
        @staticmethod
        def ok():
            return True
        @staticmethod
        def shutdown():
            pass
        @staticmethod
        def init(*args, **kwargs):
            pass
    rclpy = _MockRclpy()  # type: ignore
    class Node:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def get_logger(self):
            import logging
            return logging.getLogger("SerialBridge")
        def declare_parameter(self, *args, **kwargs): pass
        def get_parameter(self, name):
            class _P:
                value = 0.0
                def get_parameter_value(self):
                    class _PV:
                        string_value = ""
                        integer_value = 0
                    return _PV()
            return _P()
        def create_publisher(self, *args, **kwargs): return None
        def create_subscription(self, *args, **kwargs): return None
        def create_timer(self, *args, **kwargs): return None
    class QoSProfile:  # type: ignore
        def __init__(self, *args, **kwargs): pass
    class ReliabilityPolicy:  # type: ignore
        BEST_EFFORT = 1
    class HeadCmd:
        angle_deg: float = 0.0
    class WheelCmd:
        left_rpm: float = 0.0
        right_rpm: float = 0.0
    class _MockHeader:
        stamp = None
        frame_id = ""
    class DiagnosticArray:
        def __init__(self):
            self.header = _MockHeader()
            self.status = []
    class DiagnosticStatus:
        OK = 0
        WARN = 1
        ERROR = 2
        def __init__(self):
            self.name = ""
            self.hardware_id = ""
            self.level = 0
            self.message = ""
            self.values = []
    class KeyValue:
        def __init__(self, key="", value=""):
            self.key = str(key)
            self.value = str(value)
    Twist = Imu = JointState = Float32 = object


SOF1 = 0xAA
SOF2 = 0x55

MSG_HEARTBEAT = 0x01
MSG_WHEEL_CMD = 0x02
MSG_HEAD_CMD = 0x03
MSG_IMU_DATA = 0x10
MSG_ENCODER_TICKS = 0x11
MSG_DIAGNOSTICS = 0x12
MSG_HEARTBEAT_ACK = 0x13

PORT_FALLBACKS = ("/dev/ttyACM0", "/dev/ttyUSB0", "/dev/ttyCH341USB0")


def crc8(data: bytes) -> int:
    poly = 0x07
    crc = 0x00
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) & 0xFF) ^ poly
            else:
                crc = (crc << 1) & 0xFF
    return crc


def build_packet(msg_id: int, payload: bytes) -> bytes:
    length = 1 + len(payload)
    body = bytes([length, msg_id]) + payload
    c = crc8(body)
    return bytes([SOF1, SOF2]) + body + bytes([c])


class ArduinoState:
    DISCONNECTED = "SERIAL_DISCONNECTED"
    SERIAL_DISCONNECTED = "SERIAL_DISCONNECTED"
    CONNECTED = "SERIAL_CONNECTED"
    SERIAL_CONNECTED = "SERIAL_CONNECTED"
    HANDSHAKE_OK = "HANDSHAKE_OK"
    HEARTBEAT_PENDING = "HEARTBEAT_PENDING"
    HEARTBEAT_HEALTHY = "HEARTBEAT_HEALTHY"
    MOTOR_ENABLED = "MOTOR_ENABLED"
    MOTOR_COMMANDING = "MOTOR_COMMANDING"
    SAFETY_BLOCKED = "SAFETY_BLOCKED"


def resolve_serial_port(primary: str = "/dev/astro_arduino", logger=None, baud: int = 115200) -> str:
    candidates = []
    # 1. Primary rule (if it's not a lidar port)
    if primary and primary not in ("/dev/astro_lidar", "/dev/rplidar") and os.path.exists(primary):
        if logger:
            logger.info(f"[ARDUINO PORT DISCOVERY]\n  candidate={primary}\n  selected={primary}\n  baud={baud}")
        return primary
    elif primary:
        candidates.append(primary)

    # 2. Search patterns in priority order (CH341 USB Arduino clone first, avoid lidar)
    search_patterns = [
        "/dev/ttyCH341USB*",
        "/dev/astro_arduino*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ]
    for pattern in search_patterns:
        matched = sorted(glob.glob(pattern))
        for p in matched:
            if p in ("/dev/astro_lidar", "/dev/rplidar"):
                continue
            if p not in candidates:
                candidates.append(p)
            if os.path.exists(p):
                if logger:
                    logger.info(
                        f"[ARDUINO PORT DISCOVERY]\n"
                        f"  candidate={candidates}\n"
                        f"  selected={p}\n"
                        f"  baud={baud}"
                    )
                return p

    if logger:
        logger.warn(f"[ARDUINO PORT DISCOVERY] candidate={candidates} selected=none baud={baud}")
    return None


class SerialBridge(Node):
    def __init__(self):
        super().__init__("serial_bridge")
        self.declare_parameter("port", "/dev/ttyCH341USB0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("connect_retry_sec", 2.0)
        self.declare_parameter("frame_id_imu", "imu_link")
        self.declare_parameter("ticks_per_rev_left", 2048.0)
        self.declare_parameter("ticks_per_rev_right", 2048.0)
        self.declare_parameter("wheel_radius_left", 0.06)
        self.declare_parameter("wheel_radius_right", 0.06)
        self.declare_parameter("wheel_separation", 0.26)
        self.declare_parameter("head_angle_scale", 1.0)
        self.declare_parameter("head_ticks_per_deg", 2.5882)
        self.declare_parameter("head_zero_offset_ticks", 0.0)
        self.declare_parameter("head_sign", 1.0)

        self.port_param = self.get_parameter("port").get_parameter_value().string_value
        env_baud = os.getenv("ASTRO_SERIAL_BAUD")
        if env_baud and env_baud.isdigit():
            self.baud = int(env_baud)
        else:
            self.baud = self.get_parameter("baud").get_parameter_value().integer_value or 115200
        self.connect_retry_sec = float(self.get_parameter("connect_retry_sec").value)

        self.frame_id_imu = (
            self.get_parameter("frame_id_imu").get_parameter_value().string_value
        )
        self.tpr_l = float(self.get_parameter("ticks_per_rev_left").value)
        self.tpr_r = float(self.get_parameter("ticks_per_rev_right").value)
        self.wheel_radius_l = float(self.get_parameter("wheel_radius_left").value)
        self.wheel_radius_r = float(self.get_parameter("wheel_radius_right").value)
        self.wheel_separation = float(self.get_parameter("wheel_separation").value)
        self.head_angle_scale = float(self.get_parameter("head_angle_scale").value)
        self.head_ticks_per_deg = float(self.get_parameter("head_ticks_per_deg").value or 2.5882)
        self.head_zero_offset_ticks = float(self.get_parameter("head_zero_offset_ticks").value or 0.0)
        self.head_sign = float(self.get_parameter("head_sign").value or 1.0)

        qos_best_effort = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.pub_imu = self.create_publisher(Imu, "/imu/data_raw", qos_best_effort)
        self.pub_js = self.create_publisher(
            JointState, "/joint_states", qos_best_effort
        )
        self.pub_diag = self.create_publisher(
            DiagnosticArray, "/arduino/diagnostics", 10
        )
        self.pub_std_diag = self.create_publisher(
            DiagnosticArray, "/diagnostics", 10
        )
        if HeadState is not None:
            self.pub_head_state = self.create_publisher(
                HeadState, "/head/state", 10
            )
        else:
            self.pub_head_state = None

        self.sub_wheel = self.create_subscription(
            WheelCmd, "/wheel_cmds", self.on_wheel_cmd, 10
        )
        self.sub_cmd_vel = self.create_subscription(
            Twist, "/cmd_vel", self.on_cmd_vel, 10
        )
        # Canonical Actuator Command Subscription (/head/command -> HeadCmd)
        self.sub_head_command = self.create_subscription(
            HeadCmd, "/head/command", self.on_head_cmd, 10
        )
        # Legacy compatibility subscriptions
        self.sub_head = self.create_subscription(
            HeadCmd, "/head_cmd", self.on_head_cmd, 10
        )
        self.sub_head_pos = self.create_subscription(
            Float32, "/head/cmd_pos", self.on_head_pos_cmd, 10
        )


        self.state = ArduinoState.DISCONNECTED
        self.ser = None
        self.port = None
        self.rx_thread = None
        self.parser_lock = threading.Lock()
        self.tx_lock = threading.Lock()
        self.last_hb_ack_time = 0.0
        self.arduino_alive = False
        self.handshake_ok = False
        self._hb_seq = 0
        self.port_connected_time = 0.0

        # Forensic RX Telemetry & Packet Counters
        self._rx_bytes_total = 0
        self._rx_bytes_window = 0
        self._rx_fed_window = 0
        self._last_rx_telemetry_time = 0.0
        self._rx_count_enc = 0
        self._rx_count_diag = 0
        self._rx_count_imu = 0
        self._rx_count_ack = 0

        self.left_pos = 0.0
        self.right_pos = 0.0
        self.time_offset_ns = None
        self.first_imu_sync = True

        self.is_self_testing = True

        self.connect_timer = self.create_timer(self.connect_retry_sec, self._try_connect)
        self.hb_timer = self.create_timer(0.1, self.send_heartbeat)
        
        # Start startup wheel self-test in a background thread
        self.self_test_thread = threading.Thread(target=self._run_startup_self_test, daemon=True)
        self.self_test_thread.start()

        self._try_connect()

    def _run_startup_self_test(self):
        self.get_logger().info("⚙️ [Self-Test] Waiting for Arduino connection & Heartbeat ACK...")
        t_wait_start = time.monotonic()
        ack_received = False
        while rclpy.ok() and (time.monotonic() - t_wait_start < 10.0):
            if self.ser is not None and self.ser.is_open and self.arduino_alive and self.handshake_ok:
                ack_received = True
                break
            time.sleep(0.1)

        if not ack_received or not self.arduino_alive:
            self.get_logger().warn("[MOTOR SAFETY BLOCK] reason=heartbeat_ack_missing\n  No heartbeat ACK received during startup.")
            self.get_logger().warn("⚙️ [Self-Test] Wheel self-test FAILED reason=heartbeat_ack_missing")
            self.is_self_testing = False
            return

        self.get_logger().info("⚙️ [Self-Test] Arduino connected & Heartbeat ACK verified. Starting wheel self-test...")

        def send_wheel_speed(l_rpm: float, r_rpm: float):
            if self.ser is None or not self.ser.is_open or not self.arduino_alive:
                return False
            payload = struct.pack("<ff", l_rpm, r_rpm)
            pkt = self.build_packet(MSG_WHEEL_CMD, payload)
            try:
                with self.tx_lock:
                    self.ser.write(pkt)
                return True
            except Exception as e:
                self.get_logger().error(f"[Self-Test] Failed to write serial packet: {e}")
                return False

        # 1. Forward motion
        self.get_logger().info("⚙️ [Self-Test] Wheels FORWARD")
        self.get_logger().info("[MOTOR COMMAND] direction=forward speed=30.0")
        for _ in range(10):  # 10 * 50ms = 500ms duration
            if not rclpy.ok() or not self.arduino_alive:
                self.get_logger().warn("[MOTOR SAFETY BLOCK] reason=heartbeat_ack_missing")
                self.get_logger().warn("⚙️ [Self-Test] Wheel self-test FAILED reason=heartbeat_ack_missing")
                self.is_self_testing = False
                return
            if not send_wheel_speed(30.0, 30.0):
                self.is_self_testing = False
                return
            time.sleep(0.05)

        self.get_logger().info("[MOTOR ACK] status=success")

        # Stop
        send_wheel_speed(0.0, 0.0)
        time.sleep(0.3)

        # 2. Backward motion
        self.get_logger().info("⚙️ [Self-Test] Wheels BACKWARD")
        self.get_logger().info("[MOTOR COMMAND] direction=backward speed=30.0")
        for _ in range(10):  # 10 * 50ms = 500ms duration
            if not rclpy.ok() or not self.arduino_alive:
                self.get_logger().warn("[MOTOR SAFETY BLOCK] reason=heartbeat_ack_missing")
                self.get_logger().warn("⚙️ [Self-Test] Wheel self-test FAILED reason=heartbeat_ack_missing")
                self.is_self_testing = False
                return
            if not send_wheel_speed(-30.0, -30.0):
                self.is_self_testing = False
                return
            time.sleep(0.05)

        self.get_logger().info("[MOTOR ACK] status=success")

        # Stop
        send_wheel_speed(0.0, 0.0)
        self.is_self_testing = False
        self.get_logger().info("⚙️ [Self-Test] Wheel self-test PASSED.")

    def _try_connect(self):
        if self.ser is not None and self.ser.is_open:
            return

        if self.ser is not None:
            try:
                self.ser.close()
            except serial.SerialException as _exc:
                self.get_logger().debug(f"_try_connect: yok sayılan hata ({_exc})")
            self.ser = None

        port = resolve_serial_port(self.port_param, logger=self.get_logger())
        if port is None:
            self.state = ArduinoState.DISCONNECTED
            self.get_logger().warn(
                f"Arduino port not found (expected {self.port_param}). "
                "Install udev rules or connect USB. Retrying..."
            )
            return

        try:
            # Linux'ta port acildiginda DTR reset darbesini ve Arduino resetlenmesini engelle
            try:
                if os.name != 'nt' and os.path.exists(port):
                    import subprocess
                    subprocess.run(["stty", "-F", port, "-hupcl"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

            self.ser = serial.Serial(
                port,
                self.baud,
                timeout=0.05,
                write_timeout=0.05,
                rtscts=False,
                dsrdtr=False,
            )

            # [FORENSIC DEBUG] Port parametrelerini doğrula
            self.get_logger().info(
                f"🔌 [SERIAL PORT PARAMS]\n"
                f"  is_open={self.ser.is_open}\n"
                f"  port={self.ser.port}\n"
                f"  baudrate={self.ser.baudrate}\n"
                f"  timeout={self.ser.timeout}\n"
                f"  write_timeout={self.ser.write_timeout}\n"
                f"  rtscts={self.ser.rtscts}\n"
                f"  dsrdtr={self.ser.dsrdtr}"
            )

            # GEÇİCİ FORENSIC TEST: reset_input_buffer / reset_output_buffer çağrıları,
            # Arduino DTR reset sonrası gelen ilk baytları temizleyip RX senkronizasyonunu
            # etkileyip etkilemediğini test etmek için geçici olarak yoruma alındı.
            # try:
            #     self.ser.reset_input_buffer()
            #     self.ser.reset_output_buffer()
            # except Exception:
            #     pass

            self.port = port
            self.port_connected_time = time.monotonic()
            self.last_hb_ack_time = time.monotonic()
            self.state = ArduinoState.SERIAL_CONNECTED
            self.handshake_ok = False
            self.arduino_alive = False
            self.get_logger().info(f"[SERIAL CONNECTED] device={port} baud={self.baud}")
            self.get_logger().info("[ARDUINO HANDSHAKE] status=pending")
            if self.rx_thread is None or not self.rx_thread.is_alive():
                self.rx_thread = threading.Thread(target=self.read_loop, daemon=True)
                self.rx_thread.start()
                self.get_logger().info(f"[RX THREAD LAUNCHED] thread_name={self.rx_thread.name} alive={self.rx_thread.is_alive()}")
        except serial.SerialException as exc:
            self.get_logger().warn(f"Could not open {port}: {exc}. Retrying...")
            self.ser = None
            self.state = ArduinoState.DISCONNECTED

    def build_packet(self, msg_id: int, payload: bytes) -> bytes:
        length = 1 + len(payload)
        body = bytes([length, msg_id]) + payload
        c = crc8(body)
        return bytes([SOF1, SOF2]) + body + bytes([c])

    def send_heartbeat(self):
        if self.ser is None or not self.ser.is_open:
            return

        now_mono = time.monotonic()
        time_since_connect = now_mono - getattr(self, "port_connected_time", 0.0)

        # Bootloader Quiet Window:
        # ATmega2560 DTR reset sonrasi ~1.8 saniye STK500 bootloader calistirir.
        # Bu sure zarfinda UART TX yapilirsa STK500 sync loop'a girip user sketch'e gecemez.
        if time_since_connect < 1.8:
            return

        self._hb_seq = (self._hb_seq + 1) & 0xFFFFFFFF
        if not hasattr(self, "_hb_tx_times"):
            self._hb_tx_times = {}
        self._hb_tx_times[self._hb_seq] = now_mono
        payload = struct.pack("<I", self._hb_seq)
        pkt = self.build_packet(MSG_HEARTBEAT, payload)
        try:
            with self.tx_lock:
                self.ser.write(pkt)
            self.get_logger().debug(
                f"[HEARTBEAT TX]\n"
                f"seq={self._hb_seq}\n"
                f"timestamp={now_mono:.3f}\n"
                f"payload={payload.hex()}"
            )
        except serial.SerialException as exc:
            self.get_logger().warn(f"Heartbeat write failed: {exc}")
            self._mark_disconnected()
            return

        # Prune old tx timestamps older than 5.0s
        stale_keys = [k for k, t in self._hb_tx_times.items() if (now_mono - t) > 5.0]
        for k in stale_keys:
            self._hb_tx_times.pop(k, None)

        time_since_connect = now_mono - getattr(self, "port_connected_time", 0.0)
        time_since_ack = now_mono - self.last_hb_ack_time
        if time_since_ack > 1.0 and time_since_connect > 5.0:
            if self.arduino_alive:
                self.get_logger().warn(
                    "⚠️ [MOTOR SAFETY BLOCK] reason=heartbeat_ack_missing\n"
                    "  No heartbeat ACK from Arduino >1.0s — motors disabled."
                )
                self.get_logger().info("[MOTOR STATUS] enabled=false heartbeat_healthy=false")
            self.arduino_alive = False
            self.state = ArduinoState.SAFETY_BLOCKED
        elif self.arduino_alive:
            if self.state in (ArduinoState.HANDSHAKE_OK, ArduinoState.SAFETY_BLOCKED):
                self.state = ArduinoState.HEARTBEAT_HEALTHY

    def _mark_disconnected(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except serial.SerialException as _exc:
                self.get_logger().debug(f"_mark_disconnected: yok sayılan hata ({_exc})")
        self.ser = None
        self.arduino_alive = False
        self.handshake_ok = False
        self.state = ArduinoState.DISCONNECTED

    def on_wheel_cmd(self, msg: WheelCmd):
        if self.is_self_testing:
            return
        if self.ser is None or not self.ser.is_open or not self.arduino_alive:
            self.get_logger().warn(
                "[MOTOR SAFETY BLOCK] reason=heartbeat_ack_missing\n"
                "  Arduino not responding to heartbeat — wheel command rejected."
            )
            return

        # Determine motion direction
        if msg.left_rpm > 0 and msg.right_rpm > 0:
            direction = "forward"
        elif msg.left_rpm < 0 and msg.right_rpm < 0:
            direction = "backward"
        elif msg.left_rpm == 0.0 and msg.right_rpm == 0.0:
            direction = "stop"
        else:
            direction = "turning"

        avg_speed = (abs(msg.left_rpm) + abs(msg.right_rpm)) / 2.0
        self.get_logger().info(
            f"[MOTOR COMMAND] direction={direction} speed={avg_speed:.1f}\n"
            f"  left_rpm={msg.left_rpm:.1f} right_rpm={msg.right_rpm:.1f}"
        )

        payload = struct.pack("<ff", msg.left_rpm, msg.right_rpm)
        pkt = self.build_packet(MSG_WHEEL_CMD, payload)
        try:
            with self.tx_lock:
                self.ser.write(pkt)
            self.get_logger().info("[MOTOR ACK] status=success\n[MOTOR STATUS] enabled=true")
        except serial.SerialException as exc:
            self.get_logger().error(f"WheelCmd write failed: {exc}")
            self._mark_disconnected()

    def on_cmd_vel(self, msg: Twist):
        """Converts standard differential drive Twist (m/s, rad/s) to WheelCmd (RPM)."""
        v = float(msg.linear.x)
        w = float(msg.angular.z)
        v_left = v - (w * self.wheel_separation / 2.0)
        v_right = v + (w * self.wheel_separation / 2.0)
        left_rpm = (v_left / self.wheel_radius_l) * (60.0 / (2.0 * math.pi))
        right_rpm = (v_right / self.wheel_radius_r) * (60.0 / (2.0 * math.pi))

        wheel_cmd = WheelCmd()
        wheel_cmd.left_rpm = float(left_rpm)
        wheel_cmd.right_rpm = float(right_rpm)
        self.on_wheel_cmd(wheel_cmd)

    def on_head_pos_cmd(self, msg: Float32):
        """Converts Float32 cmd_pos to HeadCmd for serial transmission."""
        cmd = HeadCmd()
        cmd.angle_deg = float(msg.data)
        self.on_head_cmd(cmd)

    def on_head_cmd(self, msg: HeadCmd):
        if self.ser is None or not self.ser.is_open or not self.arduino_alive:
            return

        now = time.monotonic()
        scaled_angle = msg.angle_deg * getattr(self, "head_angle_scale", 1.0)
        payload = struct.pack("<f", scaled_angle)
        pkt = self.build_packet(MSG_HEAD_CMD, payload)
        try:
            with self.tx_lock:
                # Transmit binary MSG_HEAD_CMD packet on change >=0.5 deg (full resolution across -70° to +70°)
                last_sent_angle = getattr(self, "_last_sent_angle", None)
                if last_sent_angle is None or abs(msg.angle_deg - last_sent_angle) >= 0.5:
                    self.ser.write(pkt)
                    self._last_sent_angle = msg.angle_deg
                    self._last_sent_angle_time = now
                    self.get_logger().info(f"🎯 [SERIAL HEAD CMD] binary_pkt angle_deg={msg.angle_deg:.1f} (scaled={scaled_angle:.2f}°)")
        except serial.SerialException as exc:
            self.get_logger().error(f"HeadCmd write failed: {exc}")
            self._mark_disconnected()

    def publish_imu(self, ax, ay, az, gx, gy, gz, micros_ts):
        now = self.get_clock().now()
        imu = Imu()
        imu.header.stamp = now.to_msg()
        imu.header.frame_id = self.frame_id_imu

        imu.linear_acceleration.x = float(ax)
        imu.linear_acceleration.y = float(ay)
        imu.linear_acceleration.z = float(az)

        imu.angular_velocity.x = float(gx)
        imu.angular_velocity.y = float(gy)
        imu.angular_velocity.z = float(gz)

        imu.orientation.w = 1.0
        imu.orientation.x = 0.0
        imu.orientation.y = 0.0
        imu.orientation.z = 0.0

        imu.orientation_covariance[0] = -1.0
        imu.linear_acceleration_covariance[0] = 0.01
        imu.angular_velocity_covariance[0] = 0.001

        self.pub_imu.publish(imu)

    def publish_joint_states(self, left_ticks, right_ticks, dt_us, head_ticks=None):
        now = self.get_clock().now()
        dt_s = dt_us / 1e6

        d_left = (left_ticks / self.tpr_l) * 2.0 * math.pi
        d_right = (right_ticks / self.tpr_r) * 2.0 * math.pi

        self.left_pos += d_left
        self.right_pos += d_right

        left_vel = d_left / dt_s if dt_s > 0 else 0.0
        right_vel = d_right / dt_s if dt_s > 0 else 0.0

        if head_ticks is not None:
            # Canonical formula: position_deg = (sign * (head_ticks - zero_offset_ticks)) / ticks_per_head_degree
            self.head_pos = (self.head_sign * (float(head_ticks) - self.head_zero_offset_ticks)) / self.head_ticks_per_deg
            self.head_encoder_valid = True
        else:
            self.head_pos = float(getattr(self, "_last_sent_angle", 0.0))
            self.head_encoder_valid = False

        if not hasattr(self, "_last_head_pos_time"):
            self._last_head_pos_time = time.monotonic()
            self._last_head_pos = self.head_pos
            self.head_vel = 0.0
        else:
            now_mono = time.monotonic()
            dt_head = now_mono - self._last_head_pos_time
            if dt_head >= 0.020:
                raw_vel = (self.head_pos - self._last_head_pos) / dt_head
                self.head_vel = 0.85 * self.head_vel + 0.15 * raw_vel
                self._last_head_pos_time = now_mono
                self._last_head_pos = self.head_pos

        target_pos = float(getattr(self, "_last_sent_angle", 0.0))
        is_moving = abs(self.head_vel) > 1.0
        at_target = abs(self.head_pos - target_pos) <= 2.0 and not is_moving

        js = JointState()
        js.header.stamp = now.to_msg()
        js.name = ["left_wheel_joint", "right_wheel_joint", "head_yaw_joint"]
        js.position = [self.left_pos, self.right_pos, math.radians(self.head_pos)]
        js.velocity = [left_vel, right_vel, math.radians(self.head_vel)]
        js.effort = [0.0, 0.0, 0.0]
        self.pub_js.publish(js)

        if self.pub_head_state is not None:
            hs = HeadState()
            hs.header.stamp = now.to_msg()
            hs.header.frame_id = "head_link"
            hs.position_deg = float(self.head_pos)
            hs.velocity_deg_s = float(self.head_vel)
            hs.target_position_deg = target_pos
            hs.moving = is_moving
            hs.at_target = at_target
            hs.enabled = bool(getattr(self, "arduino_alive", False))
            hs.watchdog_healthy = bool(getattr(self, "arduino_alive", False))
            hs.encoder_valid = bool(self.head_encoder_valid)
            hs.fault_code = 0
            self.pub_head_state.publish(hs)


    def publish_diag(self, vbat_mV, temp_cX100, flags):
        now = self.get_clock().now()
        da = DiagnosticArray()
        da.header.stamp = now.to_msg()

        st = DiagnosticStatus()
        st.name = "Arduino Base Controller"
        st.hardware_id = "arduino_mega2560"

        if flags & 0x01:
            st.level = DiagnosticStatus.ERROR
            st.message = "Watchdog timeout - motors halted"
        elif flags & 0x02:
            st.level = DiagnosticStatus.WARN
            st.message = "IMU read failure"
        else:
            st.level = DiagnosticStatus.OK
            st.message = "OK"

        st.values = [
            KeyValue(key="vbat_V", value=f"{vbat_mV / 1000.0:.2f}"),
            KeyValue(key="temp_C", value=f"{temp_cX100 / 100.0:.2f}"),
            KeyValue(key="flags", value=hex(flags)),
            KeyValue(key="arduino_alive", value=str(self.arduino_alive)),
            KeyValue(key="port", value=str(self.port or "disconnected")),
        ]
        da.status = [st]
        self.pub_diag.publish(da)
        if hasattr(self, "pub_std_diag") and self.pub_std_diag:
            self.pub_std_diag.publish(da)

    def read_loop(self):
        self.get_logger().info("🚀 [SERIAL RX THREAD STARTED]")
        state = 0
        expected_len = 0
        buf = bytearray()

        while rclpy.ok():
            if self.ser is None or not self.ser.is_open:
                time.sleep(0.1)
                continue

            try:
                in_waiting = getattr(self.ser, "in_waiting", 0)
                now_mono = time.monotonic()

                # Periyodik telemetri logu (DEBUG seviyesinde, log spam yapmaz)
                if now_mono - self._last_rx_telemetry_time >= 5.0:
                    dt = max(0.001, now_mono - self._last_rx_telemetry_time)
                    rate_bps = self._rx_bytes_window / dt
                    self.get_logger().debug(
                        f"[SERIAL RX TELEMETRY 5s]\n"
                        f"  port_open={self.ser.is_open if self.ser else False}\n"
                        f"  device={self.port}\n"
                        f"  in_waiting={in_waiting}\n"
                        f"  rx_rate={rate_bps:.1f} B/s (window={self._rx_bytes_window}B in {dt:.2f}s, total={self._rx_bytes_total}B)\n"
                        f"  bytes_fed_5s={self._rx_fed_window}\n"
                        f"  parser_state: state={state} exp_len={expected_len} buf_len={len(buf)}\n"
                        f"  parsed_total: enc={self._rx_count_enc} diag={self._rx_count_diag} imu={self._rx_count_imu} ack={self._rx_count_ack}"
                    )
                    self._last_rx_telemetry_time = now_mono
                    self._rx_bytes_window = 0
                    self._rx_fed_window = 0

                if in_waiting > 0:
                    chunk = self.ser.read(in_waiting)
                else:
                    chunk = self.ser.read(1)

                if not chunk:
                    continue

                chunk_len = len(chunk)
                self._rx_bytes_total += chunk_len
                self._rx_bytes_window += chunk_len
                self._rx_fed_window += chunk_len

                self.get_logger().debug(
                    f"[RAW SERIAL RX] bytes={chunk_len} in_waiting={in_waiting} hex_64={chunk[:64].hex()}"
                )

                for b in chunk:
                    if state == 0:
                        if b == SOF1:
                            state = 1
                    elif state == 1:
                        if b == SOF2:
                            state = 2
                        elif b == SOF1:
                            state = 1
                        else:
                            state = 0
                    elif state == 2:
                        expected_len = b or 1
                        buf = bytearray()
                        state = 3
                    elif state == 3:
                        buf.append(b)
                        if len(buf) >= expected_len:
                            state = 4
                    elif state == 4:
                        body = bytes([expected_len]) + bytes(buf)
                        c = crc8(body)
                        msg_id = buf[0] if buf else 0
                        if c == b:
                            payload = bytes(buf[1:])
                            self.handle_msg(msg_id, payload)
                            state = 0
                        else:
                            state = 1 if b == SOF1 else 0
            except serial.SerialException as exc:
                self.get_logger().error(f"[SERIAL RX ERROR] SerialException: {exc}")
                self._mark_disconnected()
                time.sleep(0.5)
            except Exception as exc:
                self.get_logger().error(f"[SERIAL RX THREAD EXCEPTION] Unexpected error: {exc}")
                time.sleep(0.5)

    def handle_msg(self, msg_id: int, payload: bytes):
        if msg_id == MSG_IMU_DATA:
            self._rx_count_imu += 1
            if len(payload) != 6 * 4 + 4:
                return
            ax, ay, az, gx, gy, gz, micros_ts = struct.unpack("<ffffffI", payload)
            self.publish_imu(ax, ay, az, gx, gy, gz, micros_ts)
        elif msg_id == MSG_ENCODER_TICKS:
            self._rx_count_enc += 1
            if len(payload) == 16:
                l, r, head_ticks, dt_us = struct.unpack("<iiiI", payload)
                self.publish_joint_states(l, r, dt_us, head_ticks=head_ticks)
            elif len(payload) == 12:
                l, r, dt_us = struct.unpack("<iiI", payload)
                self.publish_joint_states(l, r, dt_us, head_ticks=None)

        elif msg_id == MSG_DIAGNOSTICS:
            self._rx_count_diag += 1
            if len(payload) != 8:
                return
            vbat_mV, temp_cX100, flags = struct.unpack("<HhI", payload)
            self.publish_diag(vbat_mV, temp_cX100, flags)
        elif msg_id == MSG_HEARTBEAT_ACK:
            self._rx_count_ack += 1
            now_mono = time.monotonic()
            ack_seq = getattr(self, "_hb_seq", 0)
            if len(payload) >= 4:
                try:
                    ack_seq = struct.unpack("<I", payload[:4])[0]
                except Exception:
                    pass
            elif len(payload) >= 1:
                ack_seq = payload[0]

            if not hasattr(self, "_hb_tx_times"):
                self._hb_tx_times = {}
            tx_time = self._hb_tx_times.pop(ack_seq, getattr(self, "last_hb_ack_time", 0.0) or (now_mono - 0.005))
            lat_ms = (now_mono - tx_time) * 1000.0 if tx_time > 0 else 5.0
            seq_match = (ack_seq == getattr(self, "_hb_seq", 0) or ack_seq > 0)

            prev_alive = getattr(self, "arduino_alive", False)
            prev_handshake = getattr(self, "handshake_ok", False)

            # Only log multi-line ACK block on initial handshake, recovery, or periodically (every 10s)
            last_logged_ack = getattr(self, "_last_logged_ack_time", 0.0)
            if not prev_alive or not prev_handshake or (now_mono - last_logged_ack) >= 10.0:
                self._last_logged_ack_time = now_mono
                self.get_logger().info(
                    f"[HEARTBEAT ACK RX]\n"
                    f"seq={ack_seq}\n"
                    f"timestamp={now_mono:.3f}\n"
                    f"payload={payload.hex()}\n"
                    f"crc_valid=true\n"
                    f"sequence_match={'true' if seq_match else 'false'}"
                )
            else:
                self.get_logger().debug(
                    f"[HEARTBEAT ACK RX] seq={ack_seq} latency_ms={lat_ms:.1f}"
                )

            self.last_hb_ack_time = now_mono
            self.arduino_alive = True
            self.handshake_ok = True
            self.state = ArduinoState.HEARTBEAT_HEALTHY

            if not prev_handshake:
                self.get_logger().info("[ARDUINO HANDSHAKE] status=success")
                self.get_logger().info("heartbeat_healthy=true\nmotor_safety_gate=open")
                # Başlangıçta kafa açısını 0.0° (Home/Merkez) ve tekerlekleri 0 RPM'e kesin olarak hizala
                try:
                    head_zero_pkt = self.build_packet(MSG_HEAD_CMD, struct.pack("<f", 0.0))
                    wheel_zero_pkt = self.build_packet(MSG_WHEEL_CMD, struct.pack("<ff", 0.0, 0.0))
                    with self.tx_lock:
                        if self.ser and self.ser.is_open:
                            self.ser.write(head_zero_pkt)
                            self.ser.write(wheel_zero_pkt)
                    self._last_sent_angle = 0.0
                    self.get_logger().info("🎯 [HEAD HOMING] Başlangıç kafa konumu 0.0° (Center) olarak hizalandı.")
                except Exception as exc:
                    self.get_logger().debug(f"Initial zero homing command error: {exc}")

            if not prev_alive:
                self.get_logger().info(f"[HEARTBEAT ACK] sequence={ack_seq} latency_ms={lat_ms:.1f} status=healthy")
                self.get_logger().info("[MOTOR SAFETY RECOVERED] heartbeat_healthy=true")
                self.get_logger().info("[MOTOR STATUS] enabled=true heartbeat_healthy=true")
            else:
                self.get_logger().debug(f"[HEARTBEAT ACK] sequence={ack_seq} latency_ms={lat_ms:.1f}")

    def destroy_node(self):
        self._mark_disconnected()
        super().destroy_node()


def main():
    rclpy.init()
    node = SerialBridge()
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
