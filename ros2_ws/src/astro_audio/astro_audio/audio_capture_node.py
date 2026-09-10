#!/usr/bin/env python3
"""ASTRO V1 — Audio Capture Node (ReSpeaker 16kHz Mono + Hardware VAD).

Matches TTS/hey_groq_assistant.py hardware capture pipeline:
  - Finds hardware ReSpeaker USB input device directly via sounddevice
  - Records 1-channel (mono) at 16000Hz (ReSpeaker on-board AEC + Beamformed audio)
  - VAD with 500 int16 RMS threshold
  - Publishes to /audio/speech_audio and /audio/vad
"""

import os
import logging

_LOG = logging.getLogger(__name__)

import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32, Int16MultiArray
except ImportError:
    rclpy = None
    Node = object
    Bool = Float32 = Int16MultiArray = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

from astro_audio.respeaker_usb import ReSpeakerHID


RESPEAKER_NAME_HINTS = ("respeaker", "uac1", "seeed", "arrayuac", "4 mic array", "array uac")


def list_input_devices() -> list[tuple[int, str]]:
    """Giriş yapabilen (kanal sayısı > 0) tüm ses cihazlarını döndürür."""
    if sd is None:
        return []
    try:
        return [
            (i, dev.get("name", "?"))
            for i, dev in enumerate(sd.query_devices())
            if dev.get("max_input_channels", 0) > 0
        ]
    except Exception:
        return []


def find_input_device(preferred: str = "") -> tuple[int | None, str, str]:
    """Kullanılacak mikrofonu seçer.

    Sıra: elle verilen cihaz → ReSpeaker → sistem varsayılanı.
    Dönen üçüncü değer seçimin nedenidir; log bunu olduğu gibi yazar ki
    ReSpeaker takılı değilken "ReSpeaker aktif" gibi yanıltıcı satır çıkmasın.
    """
    if sd is None:
        return None, "sounddevice kurulu değil", "none"

    inputs = list_input_devices()
    if not inputs:
        return None, "giriş yapabilen cihaz yok", "none"

    # 1. Elle seçim: indeks ("12") ya da ad parçası ("ReSpeaker", "HDA Intel")
    if preferred:
        if preferred.strip().lstrip("-").isdigit():
            index = int(preferred)
            for i, name in inputs:
                if i == index:
                    return i, name, "override"
        else:
            needle = preferred.strip().lower()
            for i, name in inputs:
                if needle in name.lower():
                    return i, name, "override"
        # Eşleşme yoksa çağıran uyarır ve otomatik seçime devam edilir.

    # 2. ReSpeaker dizisi
    for i, name in inputs:
        if any(hint in name.lower() for hint in RESPEAKER_NAME_HINTS):
            return i, name, "respeaker"

    # 3. Sistem varsayılanı (PulseAudio/PipeWire "default" da buraya düşer)
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            for i, name in inputs:
                if i == default_in:
                    return i, name, "default"
    except Exception as _exc:
        _LOG.debug("find_input_device: yok sayılan hata (%s)", _exc)

    # 4. Varsayılan da yoksa ilk giriş cihazı
    return inputs[0][0], inputs[0][1], "first"


class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__("audio_capture_node")

        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("chunk_size", 960)  # 60ms chunk (0.06s * 16000 = 960)
        # 750: ReSpeaker donanımıyla sahada ayarlanan değer (int16 RMS ölçeği)
        self.declare_parameter("vad_threshold", 750.0)
        # Mikrofonu elle sabitlemek için: indeks ("12") veya ad parçası ("ReSpeaker").
        # Boşsa ReSpeaker aranır, bulunamazsa sistem varsayılanına düşülür.
        self.declare_parameter("input_device", os.getenv("AUDIO_INPUT_DEVICE", ""))

        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.chunk_size = int(self.get_parameter("chunk_size").value)
        self.vad_threshold = float(self.get_parameter("vad_threshold").value)

        # Eşik int16 RMS ölçeğindedir (~450). Eski sürümlerdeki 0-1 arası enerji
        # oranı değerleri buraya düşerse VAD sürekli tetiklenir; erken uyar.
        if 0.0 < self.vad_threshold < 1.0:
            self.get_logger().warn(
                f"vad_threshold={self.vad_threshold} çok küçük — bu düğüm int16 RMS "
                f"ölçeği kullanıyor (tipik: 300-600). Varsayılan 450'ye çekiliyor."
            )
            self.vad_threshold = 450.0

        # Publishers
        self.pub_raw = self.create_publisher(Int16MultiArray, "audio_raw", 10)
        self.pub_speech = self.create_publisher(Int16MultiArray, "/audio/speech_audio", 10)
        self.pub_vad = self.create_publisher(Bool, "/audio/vad", 10)
        self.pub_doa = self.create_publisher(Float32, "audio/doa", 10)

        self.respeaker = ReSpeakerHID()
        self._audio_lock = threading.Lock()
        self._pending = None
        self._noise_floor = 150.0

        # Mikrofon seçimi
        preferred = str(self.get_parameter("input_device").value or "")
        dev_id, dev_name, source = find_input_device(preferred)

        if preferred and source != "override":
            self.get_logger().warn(
                f"İstenen mikrofon bulunamadı: \"{preferred}\" — otomatik seçime geçiliyor"
            )

        reason = {
            "override": "elle seçildi",
            "respeaker": "ReSpeaker dizisi bulundu",
            "default": "ReSpeaker yok, sistem varsayılan mikrofonu",
            "first": "ReSpeaker ve varsayılan yok, ilk giriş cihazı",
            "none": "kullanılabilir mikrofon yok",
        }[source]

        if source in ("default", "first"):
            self.get_logger().warn(
                f"⚠️ [AUDIO_DEVICE_FALLBACK]: ReSpeaker donanımı bulunamadı! "
                f"Seçilen cihaz: [{dev_id}] {dev_name} ({reason})"
            )

        if dev_id is None:
            self.get_logger().error(f"❌ [Mikrofon] {reason} — ses yakalama devre dışı")
            for i, name in list_input_devices():
                self.get_logger().info(f"    [{i}] {name}")
        else:
            self.get_logger().info(
                f"🎤 [Mikrofon] [{dev_id}] {dev_name} — {reason} (Mono {self.sample_rate} Hz)"
            )
            self.get_logger().debug(
                "Başka bir mikrofon için: AUDIO_INPUT_DEVICE=\"<indeks veya ad>\" | "
                f"Mevcut girişler: {list_input_devices()}"
            )

        self.stream = None
        if sd is not None and dev_id is not None:
            opened = self._open_stream(dev_id, dev_name)

            # Seçilen cihaz açılamadıysa (örn. ham ALSA cihazı 16 kHz'i reddediyor)
            # sistem varsayılanına düş: PulseAudio/PipeWire hız ve kanal dönüşümünü
            # kendisi yapar, böylece mikrofon tamamen sessiz kalmaz.
            if not opened and source != "default":
                fallback = self._default_device()
                if fallback and fallback[0] != dev_id:
                    self.get_logger().warn(
                        f"{dev_name} açılamadı — sistem varsayılanına geçiliyor: {fallback[1]}"
                    )
                    opened = self._open_stream(*fallback)

            if not opened:
                self.get_logger().error(
                    "❌ [Mikrofon] Hiçbir giriş cihazı açılamadı. Kullanılabilir cihazlar:"
                )
                for i, name in list_input_devices():
                    self.get_logger().error(f"    [{i}] {name}")

        self.create_timer(0.02, self._publish_pending)
        self.create_timer(0.1, self._publish_hid)

    @staticmethod
    def _default_device() -> tuple[int, str] | None:
        """Sistemin varsayılan giriş cihazını (indeks, ad) olarak döndürür."""
        try:
            default_in = sd.default.device[0]
            for i, name in list_input_devices():
                if i == default_in:
                    return i, name
        except Exception as _exc:
            self.get_logger().debug(f"_default_device: yok sayılan hata ({_exc})")
        return None

    def _open_stream(self, dev_id: int, dev_name: str) -> bool:
        """Cihazı açmayı dener; mono olmazsa 2 kanal dener. Başarıyı döndürür.

        ReSpeaker'ın mono kanalı zaten işlenmiş (AEC + beamform) sesi verir; sıradan
        mikrofonlar mono açılmayı reddederse 2 kanala düşülür.
        """
        for channels in (1, 2):
            try:
                self.stream = sd.InputStream(
                    device=dev_id,
                    channels=channels,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    dtype="int16",
                    callback=self._audio_callback,
                )
                self.stream.start()
                self.get_logger().info(
                    f"✅ [Mikrofon] Ses yakalama aktif ve dinliyor! ({dev_name}, {channels} kanal)"
                )
                return True
            except Exception as e:
                self.stream = None
                self.get_logger().warn(f"{dev_name} {channels} kanalda açılamadı: {e}")
        return False

    def _audio_callback(self, indata, frames, time_info, status):
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = indata[:, 0].copy()
        else:
            mono = indata.flatten().copy()

        # Calculate exact int16 RMS (same as hey_groq_assistant.py)
        rms = int(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
        peak = int(np.max(np.abs(mono))) if len(mono) > 0 else 0

        # Adaptive noise floor tracking
        if rms < self._noise_floor * 2.0:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * float(rms)

        # Dynamic VAD threshold with peak amplitude check (Real voice has peak > 1100)
        dynamic_thresh = max(self.vad_threshold, self._noise_floor + 300.0)
        is_speech = (rms > dynamic_thresh) and (peak > 1100)

        with self._audio_lock:
            self._pending = (mono.tolist(), is_speech)

    def _publish_pending(self):
        with self._audio_lock:
            pending = self._pending
            self._pending = None

        if pending is not None:
            mono, vad_active = pending

            vad_msg = Bool()
            vad_msg.data = vad_active
            self.pub_vad.publish(vad_msg)

            speech_msg = Int16MultiArray()
            speech_msg.data = mono
            self.pub_speech.publish(speech_msg)

    def _publish_hid(self):
        # Okuyucu bağlantı yokken de çağrılır: sonradan takılan USB bulunabilsin.
        angle = self.respeaker.doa_angle()
        if angle is not None:
            msg = Float32()
            msg.data = angle
            self.pub_doa.publish(msg)

    def destroy_node(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()
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