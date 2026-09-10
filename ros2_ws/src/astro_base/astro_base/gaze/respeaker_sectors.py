"""Eğik göz montajının 8 Eylül 2026 saha ölçümlerinden kaba yüz arama yönü.

Sol 45/90 ölçümleri monoton değil; gerçek kerteriz interpolasyonu yapılmaz.
Yalnızca ölçülen kümeler kabul edilir. 55° arama hedefi, 72° kamera FOV'u ile
45–90° tarafını kapsamak için seçilen politikadır; açısal doğruluk iddiası değil.
"""

import math
from typing import Optional


class ReSpeakerEyeSectors:
    """Ham DOA -> geometrik DOA (0 ön, saat yönü pozitif), üç yeni örnek ister."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._candidate = None
        self._hits = 0
        self._last_time = float("-inf")

    def update(self, raw: float, timestamp: float) -> Optional[float]:
        # Kontrol/kamera çevrimleri aynı USB örneğini yeni kanıt sayamaz.
        if timestamp <= self._last_time:
            return None
        if timestamp - self._last_time > 0.5:
            self.reset()
        self._last_time = timestamp
        candidate = None
        if math.isfinite(raw):
            if 34.0 <= raw <= 36.0 or 50.0 <= raw <= 58.0:
                candidate = 305.0  # Sol: ortak dönüşümde +55°.
            elif 61.0 <= raw <= 74.0:
                candidate = 0.0
            elif 138.0 <= raw <= 144.0 or 146.0 <= raw <= 151.0:
                candidate = 55.0   # Sağ: ortak dönüşümde -55°.
        if candidate is None:
            self._candidate = None
            self._hits = 0
            return None
        self._hits = self._hits + 1 if candidate == self._candidate else 1
        self._candidate = candidate
        return candidate if self._hits >= 3 else None
