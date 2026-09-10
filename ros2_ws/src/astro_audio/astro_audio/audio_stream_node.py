#!/usr/bin/env python3
"""ASTRO V1 — Real-Time Audio Streaming & Playback Engine for OpenAI Realtime API.

Features:
  - 16kHz hardware native capture (ReSpeaker 4-Mic USB Array) with 24kHz upsampling for OpenAI
  - Zero-latency non-blocking streaming playback of OpenAI response.audio.delta (24kHz -> 16kHz DAC)
  - Sub-millisecond queue flush & output stream abort on user barge-in (/tts/interrupt)
  - Real-time RMS acoustic monitoring & hardware auto-selection
"""

import base64
import logging

_LOG = logging.getLogger(__name__)

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32, String
except ImportError:
    rclpy = None
    class Node:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def get_logger(self):
            import logging
            return logging.getLogger("AudioStreamNode")
        def create_subscription(self, *args, **kwargs):
            return None
        def create_publisher(self, *args, **kwargs):
            return None
        def create_timer(self, *args, **kwargs):
            return None
        def declare_parameter(self, *args, **kwargs):
            return None
        def has_parameter(self, *args, **kwargs):
            return False
        def get_parameter(self, *args, **kwargs):
            class _Param:
                value = None
            return _Param()
        def destroy_node(self):
            pass
    class _MockMsg:
        data: Any = None
    Bool = Float32 = String = _MockMsg  # type: ignore

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from astro_audio.doa_estimator import AcousticDOAEstimator, ReSpeakerGeometry
except ImportError:
    try:
        from .doa_estimator import AcousticDOAEstimator, ReSpeakerGeometry
    except ImportError:
        try:
            _this_dir = os.path.dirname(os.path.abspath(__file__))
            if _this_dir not in sys.path:
                sys.path.insert(0, _this_dir)
            from doa_estimator import AcousticDOAEstimator, ReSpeakerGeometry
        except ImportError:
            AcousticDOAEstimator = None  # type: ignore
            ReSpeakerGeometry = None  # type: ignore


RESPEAKER_NAME_HINTS = ("respeaker", "uac1", "seeed", "arrayuac", "usb audio")
RESPEAKER_ALSA_DEVICE = "plughw:CARD=ArrayUAC10,DEV=0"
HW_SAMPLE_RATE = 16000  # ReSpeaker native hardware rate
TARGET_SAMPLE_RATE = 24000  # OpenAI Realtime standard
CHANNELS = 1
DTYPE = "int16"
CHUNK_MS = 20  # 20ms chunks = 320 samples @ 16kHz
HW_BLOCK_SIZE = int(HW_SAMPLE_RATE * (CHUNK_MS / 1000.0))  # 320


class ArecordStream:
    """Direct ALSA raw PCM capture via arecord subprocess.

    Used when PortAudio/sounddevice cannot enumerate ReSpeaker hardware
    or falls back to virtual pulse/default devices on Linux/Jetson.
    Provides identical callback interface as sounddevice.RawInputStream.
    """

    def __init__(
        self,
        alsa_device: str,
        channels: int,
        rate: int,
        blocksize: int,
        callback,
        logger=None,
    ):
        self.alsa_device = alsa_device
        self.channels = channels
        self.rate = rate
        self.blocksize = blocksize
        self.callback = callback
        self.logger = logger
        self.chunk_bytes = blocksize * channels * 2  # 16-bit signed = 2 bytes/sample
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.active = False
        self.last_error: str = ""

    def start(self):
        self.stop()
        cmd = [
            "arecord",
            "-D", self.alsa_device,
            "-c", str(self.channels),
            "-r", str(self.rate),
            "-f", "S16_LE",
            "-t", "raw",
            "-q",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.chunk_bytes * 4,
            )
        except FileNotFoundError:
            self.last_error = "'arecord' binary not found in system PATH"
            if self.logger:
                self.logger.error(f"❌ [ARECORD ERROR] {self.last_error}")
            return
        except Exception as exc:
            self.last_error = f"Failed to spawn arecord: {exc}"
            if self.logger:
                self.logger.error(f"❌ [ARECORD ERROR] {self.last_error}")
            return

        # Brief delay to detect immediate initialization failures (e.g. invalid device or busy)
        time.sleep(0.08)
        if self._proc.poll() is not None:
            stderr_msg = ""
            try:
                if self._proc.stderr:
                    stderr_msg = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
            except Exception:
                pass
            self.last_error = f"arecord exited immediately with code {self._proc.returncode}: {stderr_msg}"
            if self.logger:
                self.logger.error(f"❌ [ARECORD ERROR] {self.last_error}")
            self._proc = None
            return

        self._stop_event.clear()
        self.active = True
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="arecord_capture_reader"
        )
        self._thread.start()

    def _read_exact(self, n: int) -> bytes:
        chunks = []
        bytes_read = 0
        while bytes_read < n:
            if self._stop_event.is_set() or self._proc is None:
                return b""
            chunk = self._proc.stdout.read(n - bytes_read)
            if not chunk:
                return b""
            chunks.append(chunk)
            bytes_read += len(chunk)
        return b"".join(chunks)

    def _reader_loop(self):
        while not self._stop_event.is_set() and self._proc and self._proc.poll() is None:
            data = self._read_exact(self.chunk_bytes)
            if not data or len(data) < self.chunk_bytes:
                break
            try:
                # Delivers identical signature to sounddevice RawInputStream:
                # indata (bytes), frames, time_info, status
                self.callback(data, self.blocksize, None, None)
            except Exception:
                pass

        self.active = False
        try:
            if self._proc and self._proc.poll() is not None and not self._stop_event.is_set():
                stderr_msg = ""
                try:
                    if self._proc.stderr:
                        stderr_msg = self._proc.stderr.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
                self.last_error = f"arecord terminated unexpectedly (code {self._proc.returncode}): {stderr_msg}"
                if self.logger:
                    self.logger.error(f"❌ [ARECORD ERROR] {self.last_error}")
        except Exception:
            pass

    def stop(self):
        self._stop_event.set()
        self.active = False
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=0.5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
            self._thread = None

    def close(self):
        self.stop()



def resample_16k_to_24k(raw_16k_bytes: bytes) -> bytes:
    """Ultra-fast 16kHz -> 24kHz int16 PCM interpolation (320 -> 480 samples)."""
    arr_16k = np.frombuffer(raw_16k_bytes, dtype=np.int16)
    if len(arr_16k) == 0:
        return b""
    n_out = int(len(arr_16k) * 1.5)
    indices = np.linspace(0, len(arr_16k) - 1, n_out)
    arr_24k = np.interp(indices, np.arange(len(arr_16k)), arr_16k.astype(np.float32)).astype(np.int16)
    return arr_24k.tobytes()


def resample_24k_to_16k(raw_24k_bytes: bytes) -> bytes:
    """Ultra-fast 24kHz -> 16kHz int16 PCM downsampling (480 -> 320 samples)."""
    arr_24k = np.frombuffer(raw_24k_bytes, dtype=np.int16)
    if len(arr_24k) == 0:
        return b""
    n_out = int(len(arr_24k) * (2.0 / 3.0))
    indices = np.linspace(0, len(arr_24k) - 1, n_out)
    arr_16k = np.interp(indices, np.arange(len(arr_24k)), arr_24k.astype(np.float32)).astype(np.int16)
    return arr_16k.tobytes()


def list_devices():
    if sd is None:
        return []
    try:
        return [
            (i, dev.get("name", "?"), dev.get("max_input_channels", 0), dev.get("max_output_channels", 0))
            for i, dev in enumerate(sd.query_devices())
        ]
    except Exception:
        return []


# ALSA'nın yazılımda yeniden örnekleme yapan sanal cihazları. Ham "hw:x,y"
# cihazları yalnızca donanımın kendi hızlarını kabul eder; bunlar her hızı kabul eder.
ALSA_PLUG_HINTS = ("default", "pulse", "pipewire", "sysdefault")


def find_audio_device(is_input: bool = True, preferred: str = "") -> tuple[Optional[int], str]:
    if sd is None:
        return None, "sounddevice kurulu değil"

    devs = list_devices()
    valid = [(i, name) for i, name, in_ch, out_ch in devs if (in_ch > 0 if is_input else out_ch > 0)]
    if not valid:
        return None, "uygun ses cihazı bulunamadı"

    # 1. Preferred override
    if preferred:
        if preferred.strip().lstrip("-").isdigit():
            idx = int(preferred)
            for i, name in valid:
                if i == idx:
                    return i, name
        else:
            needle = preferred.strip().lower()
            for i, name in valid:
                if needle in name.lower():
                    return i, name

    # 2. ReSpeaker match
    for i, name in valid:
        if any(h in name.lower() for h in RESPEAKER_NAME_HINTS):
            return i, name

    # 3. ALSA'nın yeniden örnekleyen sanal cihazları.
    #
    # Eskiden burada doğrudan valid[0] dönülüyordu; bu makinede o
    # "HDA Intel PCH: ALC294 Analog (hw:0,0)" oluyor. Ham hw: cihazı ALSA'nın
    # plug katmanını atlar, yani YALNIZCA donanımın kendi hızlarını kabul eder.
    # ALC294 44100/48000 destekliyor, düğüm ise 16000 istiyor; sonuç her açılışta
    # "Invalid sample rate [PaErrorCode -9997]" ve mikrofon hiç açılmıyordu.
    # default/pulse/pipewire/sysdefault yazılımda yeniden örnekler, bu yüzden
    # ham donanımın önüne alınıyor. Yalnızca isim eşlemesi yapılır — cihaz
    # yoklaması yapmak testlerde süreç çökmesine yol açtığı için tercih edilmedi.
    for i, name in valid:
        if any(h in name.lower() for h in ALSA_PLUG_HINTS):
            return i, name

    # 4. Son çare: ilk uygun cihaz (eski davranış)
    return valid[0][0], valid[0][1]


from astro_audio.respeaker_usb import ReSpeakerHID


class AudioStreamNode(Node):
    """ROS 2 Node managing real-time bidirectional audio for OpenAI Realtime WebSocket."""

    def __init__(self):
        super().__init__("audio_stream_node")
        if hasattr(self, "declare_parameter"):
            try:
                self.declare_parameter("input_channels", 0)
                self.declare_parameter("enable_hid_doa", True)
            except Exception:
                pass

        self.enable_hid_doa = True
        if hasattr(self, "get_parameter"):
            try:
                p_val = self.get_parameter("enable_hid_doa").value
                if p_val is not None:
                    if isinstance(p_val, bool):
                        self.enable_hid_doa = p_val
                    elif isinstance(p_val, str):
                        self.enable_hid_doa = p_val.strip().lower() in ("true", "1", "yes")
                    else:
                        self.enable_hid_doa = bool(p_val)
            except Exception:
                pass

        # Publishers
        self.pub_input_pcm = self.create_publisher(String, "/audio/realtime_input_pcm", 20)
        self.pub_playback_active = self.create_publisher(Bool, "/audio/playback_active", 10)
        self.pub_input_level = self.create_publisher(Float32, "/audio/mic_level", 10)
        self.pub_doa = self.create_publisher(Float32, "/audio/doa", 10)
        # Bug #4 fix: companion confidence topic so social_gaze_node doesn't
        # need to hardcode 0.85 — GCC-PHAT PSR confidence is preserved here.
        self.pub_doa_confidence = self.create_publisher(Float32, "/audio/doa_confidence", 10)
        self.pub_vad = self.create_publisher(Bool, "/audio/vad", 10)

        # Hardware ReSpeaker HID & Acoustic DOA Estimator
        self._respeaker = ReSpeakerHID() if self.enable_hid_doa else None
        self._hid_status = None
        self._doa_estimator = AcousticDOAEstimator(sample_rate=HW_SAMPLE_RATE) if AcousticDOAEstimator else None
        self._capture_channels = 1
        self._mic_channel_indices: tuple[int, ...] = (0, 1, 2, 3)
        self._last_doa_angle = 0.0
        self._last_mic_speech_time = 0.0
        self._last_forensic_telemetry_time: float = 0.0
        self._last_gcc_snapshot_time: float = 0.0

        # Subscribers
        self.create_subscription(String, "/audio/realtime_output_pcm", self._on_output_pcm, 50)
        self.create_subscription(Bool, "/tts/interrupt", self._on_interrupt, 10)

        # Device selection
        pref_in = os.getenv("AUDIO_INPUT_DEVICE", "")
        pref_out = os.getenv("AUDIO_OUTPUT_DEVICE", "")
        self._in_dev_idx, in_name = find_audio_device(is_input=True, preferred=pref_in)
        self._out_dev_idx, out_name = find_audio_device(is_input=False, preferred=pref_out)
        self._in_device_name = in_name
        self._out_device_name = out_name

        # Configurable Acoustic Echo & Barge-In Parameters
        self.echo_mute_cooldown_s = float(os.getenv("ECHO_MUTE_COOLDOWN_S", "0.65"))
        self.barge_in_protection_ms = float(os.getenv("TTS_BARGE_IN_PROTECTION_MS", "350.0"))
        self.barge_in_min_rms = float(os.getenv("BARGE_IN_MIN_RMS", "1200.0"))
        self.barge_in_playback_min_rms = float(os.getenv("BARGE_IN_PLAYBACK_MIN_RMS", "4500.0"))
        self.barge_in_noise_mult = float(os.getenv("BARGE_IN_NOISE_MULTIPLIER", "3.5"))
        self.barge_in_min_peak = int(os.getenv("BARGE_IN_MIN_PEAK", "2800"))
        self.barge_in_playback_min_peak = int(os.getenv("BARGE_IN_PLAYBACK_MIN_PEAK", "14000"))
        self._ambient_rms = 120.0
        self._playback_drop_until = 0.0

        # Complete Playback & Callback State (Initialized BEFORE spawning worker thread)
        self._play_queue: queue.Queue[bytes] = queue.Queue(maxsize=500)
        self._is_playing = False
        self._last_playback_time = 0.0
        self._playback_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._total_enqueued_bytes = 0
        self._total_played_bytes = 0
        self._total_playback_bytes = 0
        self._current_gen_played_bytes = 0
        self._cancelled_gen_ids: set[int] = set()
        self._playback_burst_active = False
        self._burst_start_time = 0.0
        self._playback_worker_alive = True
        self._playback_worker_error = "none"
        self._last_input_callback_time = time.monotonic()
        self._last_output_chunk_time = 0.0
        self._last_output_envelope: Optional[dict] = None
        self._active_provenance: dict = {}
        self._input_stream = None
        self._output_stream = None
        self._input_stream_alive = False
        self._last_cb_err_log_time = 0.0
        self._callback_exception_count = 0
        self._generation_counter = 1000

        # Telemetry State Tracking
        self._current_gen_id: Optional[int] = None
        self._gen_first_audio_logged: set[int] = set()
        self._gen_first_packet_time: dict[int, float] = {}
        self._gen_audio_bytes: dict[int, int] = {}
        self._gen_played_bytes: dict[int, int] = {}
        self._gen_packets: dict[int, int] = {}
        self._gen_stream_start: dict[int, float] = {}
        self._gen_playback_start: dict[int, float] = {}

        # Background Audio Input Stream initialization
        self._start_input_stream()

        # Dedicated Playback Thread (ALSA single-stream owner)
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()

        # Playback status ticker timer
        self.create_timer(0.1, self._publish_status)

        # ReSpeaker 4-Mic HID DOA & VAD polling timer (10 Hz)
        if not self._under_pytest() and self.enable_hid_doa and self._respeaker is not None:
            self.create_timer(0.1, self._poll_respeaker_hid)

    def _poll_respeaker_hid(self):
        """Polls ReSpeaker 4-Mic hardware parameters (DOA & VAD) and publishes to ROS topics."""
        if self._respeaker is None:
            return
        try:
            is_speech = self._respeaker.speech_detected()
            doa_angle = self._respeaker.doa_angle()
            status = "ok" if is_speech is not None and doa_angle is not None else self._respeaker.last_error or "Geçersiz USB DOA/VAD yanıtı"
            if status != self._hid_status:
                self._hid_status = status
                if status == "ok":
                    self.get_logger().info("ReSpeaker USB DOA/VAD okunuyor; /audio/doa ham montaj açısıdır.")
                else:
                    self.get_logger().warning(f"ReSpeaker DOA/VAD yok: {status}; USB bağlantısı ve udev izinlerini kontrol edin. Yeniden denenecek.")

            is_active_playback = self._acoustic_playback_active()
            vad_msg = Bool()
            vad_msg.data = is_speech is True and not is_active_playback
            self.pub_vad.publish(vad_msg)

            # Primary hardware DOA source from ReSpeaker HID (Float32 topic contract preserved)
            if is_speech is True and doa_angle is not None and not is_active_playback:
                doa_msg = Float32()
                doa_msg.data = float(doa_angle)
                self.pub_doa.publish(doa_msg)
                # Politika güveni; ölçülmüş açısal doğruluk veya olasılık değildir.
                hid_conf_msg = Float32()
                hid_conf_msg.data = 0.60
                self.pub_doa_confidence.publish(hid_conf_msg)

        except Exception as exc:
            self.get_logger().debug(f"_poll_respeaker_hid error: {exc}")

    def _acoustic_playback_active(self) -> bool:
        """Açık DAC sessiz kalabilir; gerçek oynatma, kuyruk ve yankı kuyruğu esas."""
        now = time.monotonic()
        return bool(
            self._is_playing
            or not self._play_queue.empty()
            or now - self._last_playback_time < self.echo_mute_cooldown_s
            or now - self._last_output_chunk_time < self.echo_mute_cooldown_s
        )

    @staticmethod
    def _under_pytest() -> bool:
        """Test sürecinde GERÇEK ses donanımı açılmaz."""
        return (
            "PYTEST_CURRENT_TEST" in os.environ
            or "pytest" in sys.modules
            or "unittest" in sys.modules
            or os.environ.get("ASTRO_TEST_MODE", "0") in ("1", "true", "True")
        )

    def _start_input_stream(self):
        # Under pytest/unit test, avoid touching physical audio hardware
        if self._under_pytest():
            self._input_stream_alive = False
            self.get_logger().info("[TEST] Gerçek ses donanımı açılmadı (pytest).")
            return

        # Ensure no existing input stream is holding ALSA pcmC0D0c in this process
        if self._input_stream is not None:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None
            self._input_stream_alive = False

        pref_in = os.getenv("AUDIO_INPUT_DEVICE", "")
        alsa_target = pref_in if (pref_in.startswith("hw:") or pref_in.startswith("plughw:")) else RESPEAKER_ALSA_DEVICE

        # Check candidate device:
        # If candidate is pulse/default, or not a verified hardware ReSpeaker ("arrayuac"/"respeaker"),
        # direct ALSA arecord MUST be chosen to avoid PortAudio capturing 0 RMS or locking pcmC0D0c.
        in_name_lower = (self._in_device_name or "").lower()
        is_pulse_or_default = any(h in in_name_lower for h in ("pulse", "default", "pipewire", "sysdefault"))
        is_real_hw_respeaker = ("arrayuac" in in_name_lower) or ("respeaker" in in_name_lower)
        prefer_arecord = (
            pref_in.startswith("hw:")
            or pref_in.startswith("plughw:")
            or (sd is None)
            or is_pulse_or_default
            or (not is_real_hw_respeaker)
            or sys.platform.startswith("linux")
        )

        arecord_err = ""
        sd_err = ""

        # Primary path: Direct ALSA arecord subprocess capture (sounddevice NOT opened)
        if prefer_arecord:
            self.get_logger().info(
                f"🎙️ [AUDIO CAPTURE] Selecting direct ALSA arecord capture: {alsa_target} "
                f"(PortAudio candidate='{self._in_device_name}')..."
            )
            param_val = 0
            try:
                if hasattr(self, "has_parameter") and self.has_parameter("input_channels"):
                    param_val = int(self.get_parameter("input_channels").value)
            except Exception:
                param_val = 0
            env_val = int(os.getenv("AUDIO_INPUT_CHANNELS", "0"))
            pref_ch = param_val or env_val
            self._capture_channels = pref_ch if pref_ch in (1, 2, 4, 6, 8) else 6
            if self._capture_channels >= 6:
                self._mic_channel_indices = (1, 2, 3, 4)
            else:
                self._mic_channel_indices = (0, 1, 2, 3)

            arecord_stream = ArecordStream(
                alsa_device=alsa_target,
                channels=self._capture_channels,
                rate=HW_SAMPLE_RATE,
                blocksize=HW_BLOCK_SIZE,
                callback=self._input_callback,
                logger=self.get_logger(),
            )
            arecord_stream.start()

            # If arecord succeeded, sounddevice input stream is NEVER opened!
            if arecord_stream.active:
                self._input_stream = arecord_stream
                self._input_stream_alive = True
                self._in_device_name = f"ALSA ({alsa_target}) [arecord]"
                mic_map_str = (
                    "    - Channel 0: Processed Mono (Beamformed / AEC)\n"
                    "    - Channel 1: Front Mic 0 (0 deg)\n"
                    "    - Channel 2: Right Mic 1 (+90 deg)\n"
                    "    - Channel 3: Back Mic 2 (180 deg)\n"
                    "    - Channel 4: Left Mic 3 (-90 / 270 deg)\n"
                    "    - Channel 5: Playback Loopback"
                    if self._capture_channels >= 6
                    else
                    "    - Channel 0: Front Mic 0 (0 deg)\n"
                    "    - Channel 1: Right Mic 1 (+90 deg)\n"
                    "    - Channel 2: Back Mic 2 (180 deg)\n"
                    "    - Channel 3: Left Mic 3 (-90 / 270 deg)"
                )
                self.get_logger().info(
                    f"🎛️ [DEVICE FORMAT FORENSICS - ALSA DIRECT]\n"
                    f"  capture backend: arecord subprocess pipe\n"
                    f"  ALSA device: {alsa_target}\n"
                    f"  sample rate: {HW_SAMPLE_RATE} Hz\n"
                    f"  sample format: int16 (S16_LE)\n"
                    f"  channel count: {self._capture_channels}\n"
                    f"  period/frame size: {HW_BLOCK_SIZE} samples ({(HW_BLOCK_SIZE / HW_SAMPLE_RATE) * 1000.0:.1f} ms)\n"
                    f"  chunk bytes: {HW_BLOCK_SIZE * self._capture_channels * 2} bytes\n"
                    f"  selected mic channels: {self._mic_channel_indices}\n"
                    f"  channel mapping:\n{mic_map_str}\n"
                    f"  spatial_doa_engine=ReSpeaker HID Hardware DOA + AcousticDOAEstimator"
                )
                self.get_logger().info(
                    f"🔊 [AUDIO READY]\n"
                    f"  input_device={self._in_device_name}\n"
                    f"  input_callback=alive\n"
                    f"  audio_input_callback_alive=True"
                )
                return
            else:
                arecord_err = arecord_stream.last_error
                self.get_logger().warn(
                    f"[AUDIO WARN] Direct ALSA arecord capture could not start ({arecord_err}). "
                    f"Falling back to sounddevice / PortAudio path..."
                )

        # Secondary path: sounddevice RawInputStream
        if sd is not None and self._in_dev_idx is not None:
            try:
                max_in_ch = 1
                try:
                    dev_info = sd.query_devices(self._in_dev_idx) if (sd and self._in_dev_idx is not None) else {}
                    max_in_ch = dev_info.get("max_input_channels", 1) if isinstance(dev_info, dict) else 1
                except Exception:
                    max_in_ch = 1

                param_val = 0
                try:
                    if hasattr(self, "has_parameter") and self.has_parameter("input_channels"):
                        param_val = int(self.get_parameter("input_channels").value)
                except Exception:
                    param_val = 0
                env_val = int(os.getenv("AUDIO_INPUT_CHANNELS", "0"))
                pref_ch = param_val or env_val

                if pref_ch in (1, 2, 4, 6, 8):
                    self._capture_channels = pref_ch
                elif max_in_ch >= 6:
                    self._capture_channels = 6
                elif max_in_ch >= 4:
                    self._capture_channels = 4
                else:
                    self._capture_channels = 1

                if self._capture_channels >= 6:
                    self._mic_channel_indices = (1, 2, 3, 4)
                else:
                    self._mic_channel_indices = (0, 1, 2, 3)

                self._input_stream = sd.RawInputStream(
                    samplerate=HW_SAMPLE_RATE,
                    blocksize=HW_BLOCK_SIZE,
                    device=self._in_dev_idx,
                    channels=self._capture_channels,
                    dtype=DTYPE,
                    callback=self._input_callback,
                )
                self._input_stream.start()
                self._input_stream_alive = True

                mic_map_str = (
                    "    - Channel 0: Processed Mono (Beamformed / AEC)\n"
                    "    - Channel 1: Front Mic 0 (0 deg)\n"
                    "    - Channel 2: Right Mic 1 (+90 deg)\n"
                    "    - Channel 3: Back Mic 2 (180 deg)\n"
                    "    - Channel 4: Left Mic 3 (-90 / 270 deg)\n"
                    "    - Channel 5: Playback Loopback"
                    if self._capture_channels >= 6
                    else
                    "    - Channel 0: Front Mic 0 (0 deg)\n"
                    "    - Channel 1: Right Mic 1 (+90 deg)\n"
                    "    - Channel 2: Back Mic 2 (180 deg)\n"
                    "    - Channel 3: Left Mic 3 (-90 / 270 deg)"
                )
                hostapi_info = "?"
                if sd and isinstance(dev_info, dict) and "hostapi" in dev_info:
                    try:
                        hostapi_info = sd.query_hostapis(dev_info["hostapi"]).get("name", "?")
                    except Exception:
                        hostapi_info = str(dev_info.get("hostapi", "?"))

                self.get_logger().info(
                    f"🎛️ [DEVICE FORMAT FORENSICS - SOUNDDEVICE]\n"
                    f"  ALSA device index: {self._in_dev_idx}\n"
                    f"  detected USB device name: \"{self._in_device_name}\"\n"
                    f"  hw/card: \"{dev_info.get('name', self._in_device_name) if isinstance(dev_info, dict) else self._in_device_name}\"\n"
                    f"  host API: {hostapi_info}\n"
                    f"  sample rate: {HW_SAMPLE_RATE} Hz\n"
                    f"  sample format: int16 (16-bit signed PCM, 2 bytes/sample)\n"
                    f"  channel count: {self._capture_channels} (hardware max: {max_in_ch})\n"
                    f"  period/frame size: {HW_BLOCK_SIZE} samples ({(HW_BLOCK_SIZE / HW_SAMPLE_RATE) * 1000.0:.1f} ms)\n"
                    f"  selected mic channels: {self._mic_channel_indices}\n"
                    f"  channel mapping:\n{mic_map_str}\n"
                    f"  spatial_doa_engine=ReSpeaker HID Hardware DOA + AcousticDOAEstimator"
                )
                self.get_logger().info(
                    f"🔊 [AUDIO READY]\n"
                    f"  input_device=[{self._in_dev_idx}] {self._in_device_name}\n"
                    f"  input_callback=alive\n"
                    f"  audio_input_callback_alive=True"
                )
                return
            except Exception as e:
                sd_err = str(e)
                self.get_logger().warn(
                    f"[AUDIO WARN] sounddevice capture failed on device [{self._in_dev_idx}] {self._in_device_name}: {e}"
                )
                if not prefer_arecord:
                    self.get_logger().info(f"Attempting fallback direct ALSA arecord capture: {alsa_target}...")
                    arecord_stream = ArecordStream(
                        alsa_device=alsa_target,
                        channels=self._capture_channels,
                        rate=HW_SAMPLE_RATE,
                        blocksize=HW_BLOCK_SIZE,
                        callback=self._input_callback,
                        logger=self.get_logger(),
                    )
                    arecord_stream.start()
                    if arecord_stream.active:
                        self._input_stream = arecord_stream
                        self._input_stream_alive = True
                        self._in_device_name = f"ALSA ({alsa_target}) [arecord fallback]"
                        self.get_logger().info(
                            f"🔊 [AUDIO READY]\n"
                            f"  input_device={self._in_device_name}\n"
                            f"  input_callback=alive\n"
                            f"  audio_input_callback_alive=True"
                        )
                        return
                    else:
                        arecord_err = arecord_stream.last_error
        else:
            if sd is None:
                sd_err = "sounddevice library is not installed"
            elif self._in_dev_idx is None:
                sd_err = "no valid sounddevice input device index"

        # If both direct ALSA arecord and sounddevice failed:
        self._input_stream_alive = False
        self.get_logger().error(
            f"❌ [AUDIO ERROR]\n"
            f"  direction=input\n"
            f"  device=[{self._in_dev_idx}] {self._in_device_name}\n"
            f"  reason=capture_unavailable\n"
            f"  arecord_error={arecord_err or 'not_run'}\n"
            f"  sounddevice_error={sd_err or 'not_run'}"
        )
        # Subscribe to audio_capture_node's /audio/speech_audio as fallback input transport
        try:
            from std_msgs.msg import Int16MultiArray
            self.sub_fallback_audio = self.create_subscription(
                Int16MultiArray, "/audio/speech_audio", self._on_fallback_audio_msg, 20
            )
        except Exception:
            pass

    def _on_fallback_audio_msg(self, msg):
        """Receives 16kHz int16 PCM from audio_capture_node when direct hardware capture is occupied."""
        try:
            raw_bytes = np.array(msg.data, dtype=np.int16).tobytes()
            self._process_raw_audio_chunk(raw_bytes)
        except Exception:
            pass

    def _input_callback(self, indata, frames, time_info, status):
        """Audio hardware callback triggered every 20ms with 320 16-bit PCM samples."""
        self._last_input_callback_time = time.monotonic()
        raw_bytes = bytes(indata) if indata is not None else b""
        self._process_raw_audio_chunk(raw_bytes)

    def _process_raw_audio_chunk(self, raw_bytes: bytes):
        """Processes 16kHz int16 PCM chunk (VAD, multi-channel GCC-PHAT DOA, 24kHz upsampling)."""
        try:
            if not raw_bytes:
                return

            now = time.monotonic()
            if now < self._playback_drop_until:
                return

            # Multi-channel or Mono audio unpacking
            raw_arr = np.frombuffer(raw_bytes, dtype=np.int16)
            if self._capture_channels >= 4 and len(raw_arr) >= (HW_BLOCK_SIZE * self._capture_channels):
                multi_ch = raw_arr.reshape(-1, self._capture_channels).T  # Shape: (channels, frames)
                # On 6-channel ReSpeaker:
                # ch0 is the XMOS DSP beamformed output, which heavily attenuates voice when off-axis.
                # ch1 is the true physical Front Microphone (Mic 0, 0 deg).
                # Default to ch1 (or AUDIO_SPEECH_CHANNEL override) for loud, unattenuated speech recognition.
                if self._capture_channels >= 6:
                    speech_ch = int(os.getenv("AUDIO_SPEECH_CHANNEL", "1"))
                    if speech_ch >= multi_ch.shape[0]:
                        speech_ch = 1
                else:
                    speech_ch = int(os.getenv("AUDIO_SPEECH_CHANNEL", "0"))
                    if speech_ch >= multi_ch.shape[0]:
                        speech_ch = 0
                arr = multi_ch[speech_ch]
                mono_raw_bytes = arr.tobytes()
            else:
                multi_ch = None
                arr = raw_arr
                mono_raw_bytes = raw_bytes

            # Measure RMS level and peak for diagnostic & VAD
            peak = 0
            try:
                rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                if len(arr) > 0:
                    peak = int(np.max(np.abs(arr)))
            except Exception:
                rms = 0.0
                peak = 0

            # STEP 2, 3, 5: RAW CHANNEL TELEMETRY & DISTINCTNESS FORENSICS (~1 Hz)
            if multi_ch is not None and (now - getattr(self, "_last_forensic_telemetry_time", 0.0)) >= 1.0:
                self._last_forensic_telemetry_time = now
                num_ch = multi_ch.shape[0]

                # STEP 2: Channel metrics (RMS, peak, mean, zero-crossing rate)
                ch_lines = []
                for c in range(num_ch):
                    c_f = multi_ch[c].astype(np.float32)
                    c_rms = float(np.sqrt(np.mean(c_f ** 2)))
                    c_peak = int(np.max(np.abs(multi_ch[c])))
                    c_mean = float(np.mean(c_f))
                    c_zcr = float(np.mean(np.diff(np.signbit(c_f)) != 0)) if len(c_f) > 1 else 0.0
                    ch_lines.append(f"  ch{c} rms={c_rms:7.1f} peak={c_peak:5d} mean={c_mean:+6.1f} zcr={c_zcr:.3f}")
                ch_metrics_str = "\n".join(ch_lines)

                # STEP 5: Channel distinctness normalized correlation among channels 1..4
                if num_ch >= 5:
                    def _norm_corr(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
                        xf = x_arr.astype(np.float32)
                        yf = y_arr.astype(np.float32)
                        xd = xf - np.mean(xf)
                        yd = yf - np.mean(yf)
                        denom = float(np.sqrt(np.sum(xd ** 2) * np.sum(yd ** 2)))
                        return float(np.sum(xd * yd) / denom) if denom > 1e-6 else 0.0

                    c12 = _norm_corr(multi_ch[1], multi_ch[2])
                    c13 = _norm_corr(multi_ch[1], multi_ch[3])
                    c14 = _norm_corr(multi_ch[1], multi_ch[4])
                    c23 = _norm_corr(multi_ch[2], multi_ch[3])
                    c24 = _norm_corr(multi_ch[2], multi_ch[4])
                    c34 = _norm_corr(multi_ch[3], multi_ch[4])
                    corr_str = (
                        f"  corr(ch1,ch2)={c12:.4f} corr(ch1,ch3)={c13:.4f} corr(ch1,ch4)={c14:.4f}\n"
                        f"  corr(ch2,ch3)={c23:.4f} corr(ch2,ch4)={c24:.4f} corr(ch3,ch4)={c34:.4f}"
                    )
                else:
                    corr_str = "  (fewer than 5 channels available for 1..4 pair correlation)"

                # STEP 3: Channel shape inspection
                self.get_logger().info(
                    f"[CHANNEL_SHAPE] pcm_shape={multi_ch.shape} capture_channels={self._capture_channels} selected_channels={getattr(self, '_mic_channel_indices', (0, 1, 2, 3))}\n"
                    f"RAW_AUDIO:\n"
                    f"{ch_metrics_str}\n"
                    f"CHANNEL_CORRELATIONS:\n"
                    f"{corr_str}"
                )

            is_active_playback = self._acoustic_playback_active()

            if not is_active_playback and rms < 400.0:
                # Continuously adapt ambient background noise floor during quiet periods
                self._ambient_rms = 0.96 * self._ambient_rms + 0.04 * rms
            elif not is_active_playback and rms >= 400.0:
                self._last_mic_speech_time = now

            # Multi-Channel GCC-PHAT DOA Spatial Estimation (Primary high-precision acoustic tracking)
            if multi_ch is not None and self._doa_estimator and not is_active_playback and rms >= 300.0:
                mic_indices = getattr(self, "_mic_channel_indices", (0, 1, 2, 3))
                if multi_ch.shape[0] > max(mic_indices):
                    mics = multi_ch[list(mic_indices)]
                else:
                    mics = multi_ch[:4]

                # STEP 7: GCC-PHAT INPUT SNAPSHOT (~1 Hz rate-limited)
                if (now - getattr(self, "_last_gcc_snapshot_time", 0.0)) >= 1.0:
                    self._last_gcc_snapshot_time = now
                    mics_f = mics.astype(np.float32)
                    mics_rms = [round(float(np.sqrt(np.mean(mics_f[i] ** 2))), 1) for i in range(mics.shape[0])]
                    self.get_logger().info(
                        f"[GCC_PHAT_SNAPSHOT]\n"
                        f"  selected_pcm_shape={mics.shape}\n"
                        f"  selected_channel_indices={list(mic_indices)}\n"
                        f"  selected_channel_rms={mics_rms}"
                    )

                azimuth_deg, conf, valid = self._doa_estimator.estimate_from_multichannel_pcm(mics)

            # Software Echo Mute & Self-Voice Suppression (Zero Self-Hearing):
            if is_active_playback:
                burst_start = getattr(self, "_burst_start_time", 0.0)
                if self._playback_burst_active and burst_start > 0.0 and ((now - burst_start) * 1000.0 < self.barge_in_protection_ms):
                    return

                # Target barge-in threshold during active playback: Requires intentional voice exceeding loudspeaker playback level
                playback_barge_rms = float(getattr(self, "barge_in_playback_min_rms", 4500.0))
                playback_barge_peak = int(getattr(self, "barge_in_playback_min_peak", 14000))
                adaptive_barge_in_rms = max(playback_barge_rms, self._ambient_rms * self.barge_in_noise_mult)

                # Channel correlation check: If all mic channels are highly correlated (internal speaker echo), suppress
                if multi_ch is not None and multi_ch.shape[0] >= 5:
                    def _corr(a: np.ndarray, b: np.ndarray) -> float:
                        af = a.astype(np.float32) - float(np.mean(a))
                        bf = b.astype(np.float32) - float(np.mean(b))
                        d = float(np.sqrt(np.sum(af ** 2) * np.sum(bf ** 2)))
                        return float(np.sum(af * bf) / d) if d > 1e-6 else 0.0

                    c12 = _corr(multi_ch[1], multi_ch[2])
                    c13 = _corr(multi_ch[1], multi_ch[3])
                    c14 = _corr(multi_ch[1], multi_ch[4])
                    if min(c12, c13, c14) >= 0.90:
                        return
                    if multi_ch.shape[0] >= 6:
                        ch5_rms = float(np.sqrt(np.mean(multi_ch[5].astype(np.float32) ** 2)))
                        if ch5_rms > 100.0 and _corr(multi_ch[speech_ch], multi_ch[5]) >= 0.70:
                            return

                # 2. Distinguish loud speech energy during active playback
                is_genuine_barge_in = (rms >= adaptive_barge_in_rms and peak >= playback_barge_peak)
                if not is_genuine_barge_in:
                    return

            # Energy gate: Only stream frames with meaningful speech energy.
            # Streaming continuous silence wastes bandwidth and burns TPM quota on OpenAI Realtime.
            # Gate threshold: above dead silence floor (10) AND either above ambient*0.5 or above hard minimum of 40.
            if not is_active_playback and rms < max(40.0, self._ambient_rms * 0.50):
                return

            # Publish mic level
            lvl_msg = Float32()
            lvl_msg.data = float(rms)
            self.pub_input_level.publish(lvl_msg)

            # Resample 16kHz -> 24kHz for OpenAI Realtime API (Channel 0 / Front Speech)
            pcm_24k = resample_16k_to_24k(mono_raw_bytes)

            # Encode to base64 and publish to ROS 2 topic
            b64_str = base64.b64encode(pcm_24k).decode("ascii")
            msg = String()
            msg.data = b64_str
            self.pub_input_pcm.publish(msg)
        except Exception as exc:
            self._callback_exception_count += 1
            now_cb = time.monotonic()
            if (now_cb - self._last_cb_err_log_time) > 2.0:
                self._last_cb_err_log_time = now_cb
                self.get_logger().debug(f"Input processing exception: {exc}")
                self.get_logger().error(
                    f"❌ [Realtime Audio Callback Error]: callback_exception={type(exc).__name__}: {exc} | "
                    f"callback_exception_count={self._callback_exception_count} | audio_input_alive=True"
                )

    def _on_output_pcm(self, msg: String):
        """Incoming 24kHz PCM audio chunk from OpenAI Realtime API or Fallback TTS (base64 or JSON)."""
        if not msg.data:
            return
        try:
            payload = {}
            raw_str = msg.data.strip()
            if raw_str.startswith("{") and raw_str.endswith("}"):
                try:
                    payload = json.loads(raw_str)
                    b64_pcm = payload.get("pcm") or payload.get("data", "")
                except Exception:
                    b64_pcm = raw_str
            else:
                b64_pcm = raw_str

            is_done = bool(payload.get("is_done", False))
            is_first = bool(payload.get("is_first", False))
            gen_id = payload.get("generation_id", 0)

            raw_16k = b""
            if b64_pcm:
                raw_24k = base64.b64decode(b64_pcm.encode("ascii"))
                if raw_24k:
                    # Resample 24kHz -> 16kHz for hardware ReSpeaker DAC
                    raw_16k = resample_24k_to_16k(raw_24k)

            if raw_16k or is_done:
                item = {
                    "pcm": raw_16k,
                    "generation_id": gen_id,
                    "is_first": is_first,
                    "is_done": is_done,
                    "tts_provider": payload.get("tts_provider", "openai"),
                    "tts_model": payload.get("tts_model", "gpt-realtime-2.1-mini"),
                    "tts_source": payload.get("tts_source", "realtime_openai"),
                    "playback_source": payload.get("playback_source", payload.get("tts_source", "realtime_openai")),
                }
                self._play_queue.put_nowait(item)
                self._last_output_chunk_time = time.monotonic()
                self._last_output_envelope = item
                self._total_enqueued_bytes += len(raw_16k)
        except (queue.Full, Exception) as e:
            self.get_logger().debug(f"PCM enqueue notice: {e}")

    def _on_interrupt(self, msg: Bool):
        """Zero-latency barge-in signal: instantly flush playback buffer queue and mute lingering tail."""
        try:
            if not msg.data:
                return

            # P0-7: Barge-in is only valid if playback has actually started and played bytes > 0
            if not self._playback_burst_active or self._total_played_bytes == 0:
                return

            now_mono = time.monotonic()
            barge_in_after_ms = int((now_mono - self._burst_start_time) * 1000.0) if self._burst_start_time > 0 else 0
            if self._burst_start_time > 0 and barge_in_after_ms < int(self.barge_in_protection_ms):
                self.get_logger().debug(f"🛡️ [Acoustic Gate] Interruption rejected: {barge_in_after_ms}ms < {self.barge_in_protection_ms}ms (self-voice echo)")
                return

            discarded_bytes = 0
            with self._playback_lock:
                while not self._play_queue.empty():
                    try:
                        c = self._play_queue.get_nowait()
                        raw_len = len(c["pcm"]) if isinstance(c, dict) else len(c)
                        discarded_bytes += raw_len
                    except queue.Empty:
                        break

            barge_in_source = "user"
            self._is_playing = False
            self._playback_burst_active = False
            self._last_playback_time = 0.0
            self._playback_drop_until = now_mono + 0.15
            prov = getattr(self, "_active_provenance", {})
            cancelled_gen = prov.get('generation_id', 0)
            if not hasattr(self, "_cancelled_gen_ids"):
                self._cancelled_gen_ids = set()
            self._cancelled_gen_ids.add(cancelled_gen)

            gen_bytes = getattr(self, "_current_gen_played_bytes", self._total_played_bytes)

            self.get_logger().info(
                f"⚡ [Playback Telemetry]: tts_playback_cancelled=True | "
                f"generation_id={cancelled_gen} | "
                f"playback_source={prov.get('playback_source', 'unknown')} | "
                f"tts_provider={prov.get('tts_provider', 'unknown')} | "
                f"tts_model={prov.get('tts_model', 'unknown')} | "
                f"tts_played_bytes={gen_bytes} | "
                f"tts_remaining_bytes={discarded_bytes} | "
                f"total_playback_bytes={self._total_played_bytes} | "
                f"playback_duration_ms={barge_in_after_ms} | "
                f"barge_in_after_ms={barge_in_after_ms} | "
                f"barge_in_source={barge_in_source} | "
                f"reason=barge_in"
            )
        except Exception as exc:
            self.get_logger().debug(f"_on_interrupt error: {exc}")

    def _playback_worker(self):
        """Dedicated real-time audio playback loop sending PCM directly to hardware DAC."""
        if sd is None:
            self._playback_worker_alive = False
            self._playback_worker_error = "sounddevice_library_missing"
            return

        out_stream = None
        try:
            out_stream = sd.RawOutputStream(
                samplerate=HW_SAMPLE_RATE,
                blocksize=0,
                device=self._out_dev_idx,
                channels=CHANNELS,
                dtype=DTYPE,
            )
            out_stream.start()
            self._output_stream = out_stream
            self._playback_worker_alive = True
            self._playback_worker_error = "none"
        except Exception as e:
            self._playback_worker_alive = False
            self._playback_worker_error = f"dac_init_failed: {e}"
            self.get_logger().error(
                f"❌ [Realtime Audio] Çıkış akışı başlatılamadı ({self._out_device_name}): {e} | "
                f"tts_playback_started=False | tts_playback_error={self._playback_worker_error}"
            )
            return

        active_gen_id = None
        gen_started = False
        gen_done_seen = False
        gen_start_time = 0.0
        gen_played_bytes = 0
        gen_prov = {}

        if not hasattr(self, "_cancelled_gen_ids"):
            self._cancelled_gen_ids = set()

        while not self._stop_event.is_set():
            try:
                item = self._play_queue.get(timeout=0.05)
                if isinstance(item, dict):
                    chunk = item["pcm"]
                    gen_id = item.get("generation_id", 0)
                    is_done = item.get("is_done", False)
                    tts_provider = item.get("tts_provider", "openai")
                    tts_model = item.get("tts_model", "unknown")
                    tts_source = item.get("tts_source", "unknown")
                    playback_source = item.get("playback_source", tts_source)
                else:
                    chunk = item
                    gen_id = 0
                    is_done = False
                    tts_provider = "openai"
                    tts_model = os.getenv("REALTIME_MODEL", "gpt-realtime-2.1-mini")
                    tts_source = "realtime_openai"
                    playback_source = "realtime_openai"

                # Check if this is a new generation
                if (gen_id != active_gen_id) or (not gen_started):
                    if gen_started and active_gen_id is not None and active_gen_id not in self._cancelled_gen_ids:
                        # Close prior generation cleanly if not cancelled
                        burst_dur_ms = (time.monotonic() - gen_start_time) * 1000.0
                        self.get_logger().info(
                            f"🔊 [Playback Telemetry]: tts_playback_finished=True | "
                            f"generation_id={active_gen_id} | "
                            f"playback_source={gen_prov.get('playback_source', 'unknown')} | "
                            f"tts_provider={gen_prov.get('tts_provider', 'unknown')} | "
                            f"tts_model={gen_prov.get('tts_model', 'unknown')} | "
                            f"tts_played_bytes={gen_played_bytes} | "
                            f"total_playback_bytes={self._total_played_bytes} | "
                            f"playback_duration_ms={int(burst_dur_ms)}"
                        )
                    active_gen_id = gen_id
                    gen_started = True
                    gen_done_seen = False
                    gen_start_time = time.monotonic()
                    gen_played_bytes = 0
                    self._current_gen_played_bytes = 0
                    gen_prov = {
                        "generation_id": gen_id,
                        "playback_source": playback_source,
                        "tts_provider": tts_provider,
                        "tts_model": tts_model,
                        "tts_source": tts_source,
                    }
                    self._playback_burst_active = True
                    self._burst_start_time = gen_start_time
                    self._active_provenance = gen_prov
                    self.get_logger().info(
                        f"🔊 [Playback Telemetry]: tts_playback_started=True | "
                        f"generation_id={gen_id} | playback_source={playback_source} | "
                        f"tts_provider={tts_provider} | tts_model={tts_model} | "
                        f"tts_source={tts_source} | audio_bytes={len(chunk)} | "
                        f"tts_audio_device=\"{self._out_device_name}\""
                    )

                if is_done:
                    gen_done_seen = True

                # Discard chunks if generation was cancelled by barge-in
                if active_gen_id in self._cancelled_gen_ids:
                    continue

                if chunk and len(chunk) > 0:
                    t_w_start = time.perf_counter()
                    with self._playback_lock:
                        # İlk blocking write sürerken de robot konuşuyor.
                        self._is_playing = True
                        out_stream.write(chunk)
                    t_w_end = time.perf_counter()

                    self._last_playback_time = time.monotonic()
                    gen_played_bytes += len(chunk)
                    self._current_gen_played_bytes = gen_played_bytes
                    self._total_played_bytes += len(chunk)

                # If done signal received and queue is now empty, finish generation playback
                if gen_done_seen and self._play_queue.empty():
                    self._is_playing = False
                    self._playback_burst_active = False
                    gen_started = False
                    burst_dur_ms = (time.monotonic() - gen_start_time) * 1000.0
                    if active_gen_id not in self._cancelled_gen_ids:
                        self.get_logger().info(
                            f"🔊 [Playback Telemetry]: tts_playback_finished=True | "
                            f"generation_id={active_gen_id} | "
                            f"playback_source={gen_prov.get('playback_source', 'unknown')} | "
                            f"tts_provider={gen_prov.get('tts_provider', 'unknown')} | "
                            f"tts_model={gen_prov.get('tts_model', 'unknown')} | "
                            f"tts_played_bytes={gen_played_bytes} | "
                            f"total_playback_bytes={self._total_played_bytes} | "
                            f"playback_duration_ms={int(burst_dur_ms)}"
                        )
                    active_gen_id = None

            except queue.Empty:
                if gen_started and gen_done_seen:
                    self._is_playing = False
                    self._playback_burst_active = False
                    gen_started = False
                    burst_dur_ms = (time.monotonic() - gen_start_time) * 1000.0
                    if active_gen_id not in self._cancelled_gen_ids:
                        self.get_logger().info(
                            f"🔊 [Playback Telemetry]: tts_playback_finished=True | "
                            f"generation_id={active_gen_id} | "
                            f"playback_source={gen_prov.get('playback_source', 'unknown')} | "
                            f"tts_provider={gen_prov.get('tts_provider', 'unknown')} | "
                            f"tts_model={gen_prov.get('tts_model', 'unknown')} | "
                            f"tts_played_bytes={gen_played_bytes} | "
                            f"total_playback_bytes={self._total_played_bytes} | "
                            f"playback_duration_ms={int(burst_dur_ms)}"
                        )
                    active_gen_id = None
                elif gen_started and (time.monotonic() - self._last_playback_time) > 2.0:
                    # Stream timed out without is_done (e.g. dropped connection)
                    self._is_playing = False
                    self._playback_burst_active = False
                    gen_started = False
                    burst_dur_ms = (time.monotonic() - gen_start_time) * 1000.0
                    if active_gen_id not in self._cancelled_gen_ids:
                        self.get_logger().info(
                            f"🔊 [Playback Telemetry]: tts_playback_finished=True | "
                            f"generation_id={active_gen_id} | "
                            f"playback_source={gen_prov.get('playback_source', 'unknown')} | "
                            f"tts_provider={gen_prov.get('tts_provider', 'unknown')} | "
                            f"tts_model={gen_prov.get('tts_model', 'unknown')} | "
                            f"tts_played_bytes={gen_played_bytes} | "
                            f"total_playback_bytes={self._total_played_bytes} | "
                            f"playback_duration_ms={int(burst_dur_ms)} | reason=stream_timeout"
                        )
                    active_gen_id = None
                if (time.monotonic() - self._last_playback_time) > 0.35:
                    self._is_playing = False
            except Exception as exc:
                self._is_playing = False
                self._playback_worker_error = f"{type(exc).__name__}: {exc}"
                self.get_logger().error(
                    f"❌ [Playback Worker Error]: exception={type(exc).__name__}: {exc} | "
                    f"tts_playback_started=False | tts_playback_error={self._playback_worker_error}"
                )
                time.sleep(0.05)



    def _publish_status(self):
        msg = Bool()
        msg.data = self._acoustic_playback_active()
        self.pub_playback_active.publish(msg)

    def destroy_node(self):
        self._stop_event.set()
        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:
                pass
        if self._output_stream:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioStreamNode()
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
