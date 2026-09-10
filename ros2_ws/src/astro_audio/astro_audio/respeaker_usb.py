"""ReSpeaker XVF3000 vendor-control okuyucusu; iki ROS ses yolu paylaşır.

Protokol: https://github.com/respeaker/usb_4_mic_array/blob/master/tuning.py
HID adı eski importlarla uyumluluk için korunur; aktarım HID raporu değildir.
"""

import struct
import time
from typing import Optional

try:
    import usb.core
except ImportError:
    usb = None


class ReSpeakerHID:
    TIMEOUT_MS = 1000
    RETRY_S = 5.0

    def __init__(self):
        self.dev = None
        self.last_error: Optional[str] = None
        self._last_find_attempt = float("-inf")
        self._find_device()

    def _find_device(self):
        now = time.monotonic()
        if now - self._last_find_attempt < self.RETRY_S:
            return
        self._last_find_attempt = now
        if usb is None:
            self.last_error = "pyusb kurulu değil"
            return
        try:
            self.dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            self.last_error = None if self.dev is not None else "USB 2886:0018 bulunamadı"
        except Exception as exc:
            self.last_error = str(exc)

    def _read_param(self, module_id: int, offset: int) -> Optional[int]:
        if self.dev is None:
            self._find_device()
            if self.dev is None:
                return None
        try:
            # Okuma biti | tamsayı biti | parametre ofseti; modül wIndex'te.
            data = self.dev.ctrl_transfer(
                0xC0, 0, 0xC0 | offset, module_id, 8, self.TIMEOUT_MS
            )
            if len(data) != 8:
                raise ValueError(f"Eksik USB yanıtı: {len(data)}/8 bayt")
            self.last_error = None
            return struct.unpack_from("<i", data)[0]
        except Exception as exc:
            self.last_error = str(exc)
            self.dev = None
            self._last_find_attempt = time.monotonic()
            return None

    def speech_detected(self) -> Optional[bool]:
        value = self._read_param(19, 22)  # SPEECHDETECTED; offset 0 AGCONOFF'tur.
        return bool(value) if value in (0, 1) else None

    def doa_angle(self) -> Optional[float]:
        value = self._read_param(21, 0)  # DOAANGLE
        return float(value) if value is not None and 0 <= value <= 359 else None
