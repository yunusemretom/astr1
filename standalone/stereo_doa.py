#!/usr/bin/env python3
"""İki mikrofonlu yön bulma — elde 4'lü dizi yokken.

`AcousticDOAEstimator` dört kanal ister: iki dik eksenden gecikme okur ve tam çember
üzerinde açı verir. Bu makinede öyle bir dizi yok, ama laptopun dahili çifti gerçekten
uzamsal olarak ayrı: sol hoparlörden çalınan ses -3.99 örnek, sağdan çalınan +4.19
örnek gecikme verdi (std 0.05 ve 0.03, 44.1 kHz ham aygıtta). Yani tek eksende yön
çıkarmak için yeterli bilgi var.

Tek çiftin iki sınırı var, ikisi de kabul edilmiş:

  ön/arka belirsizliği — arkadan gelen ses önden gelmiş gibi okunur. Çember üzerinde
      aynı gecikmeyi veren iki nokta var ve iki mikrofon bunları ayıramaz. Masaya
      bakan bir kafa için önemsiz; 360° kapsama isteniyorsa 4'lü dizi şart.
  ölçek mikrofon aralığına bağlı — işaret ve sıralama kesin, mutlak açı ise
      `mic_spacing_m` ne kadar doğruysa o kadar doğru. Laptop modeline göre değişir,
      bu yüzden dışarıdan verilebiliyor.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

SPEED_OF_SOUND_MPS = 343.0

# Hoparlör kalibrasyonundan türetildi: ±4.1 örnek @44.1 kHz, hoparlörler kabaca ±50°.
# Laptop modeline göre değişir; ölçüp geçmek için `mic_spacing_m` var.
DEFAULT_MIC_SPACING_M = 0.041

# PHAT tepesinin keskinliği: doğrudan yoldan gelen bir ses tek ve dar bir tepe verir,
# oda yankısı ve gürültü ise yayvan bir tabana. Bu oranın altında yön iddia edilmez.
DEFAULT_MIN_SHARPNESS = 3.0


def _gcc_phat(left: np.ndarray, right: np.ndarray, max_lag: int) -> Tuple[float, float]:
    """(gecikme, keskinlik) döndürür. Gecikme örnek cinsinden, ara-örnek çözünürlüklü.

    İşaret sözleşmesi ölçümle sabitlendi: sağdan gelen ses artı, soldan gelen eksi.
    """
    n = 1 << int(math.ceil(math.log2(len(left) + len(right))))
    spectrum = np.fft.rfft(left, n) * np.conj(np.fft.rfft(right, n))
    magnitude = np.abs(spectrum)
    magnitude[magnitude < 1e-12] = 1e-12
    correlation = np.fft.irfft(spectrum / magnitude, n)

    window = np.concatenate((correlation[-max_lag:], correlation[:max_lag + 1]))
    peak_index = int(np.argmax(window))
    peak = float(window[peak_index])

    baseline = float(np.mean(np.abs(window)))
    sharpness = peak / baseline if baseline > 0 else 0.0

    # Parabolik ara-örnek düzeltme: 44.1 kHz'de tam örnek adımı ~8°'ye denk geliyor,
    # yani yuvarlamak yönü kullanılamaz hale getirecek kadar kabalaştırıyor.
    offset = 0.0
    if 0 < peak_index < len(window) - 1:
        before, here, after = window[peak_index - 1], peak, window[peak_index + 1]
        denominator = before - 2.0 * here + after
        if denominator != 0.0:
            offset = (before - after) / (2.0 * denominator)

    return (peak_index - max_lag) + offset, sharpness


class StereoDOA:
    """Bir stereo çiftten tek eksenli kerteriz."""

    def __init__(
        self,
        sample_rate: int,
        mic_spacing_m: float = DEFAULT_MIC_SPACING_M,
        min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    ):
        self.sample_rate = int(sample_rate)
        self.mic_spacing_m = float(mic_spacing_m)
        self.min_sharpness = float(min_sharpness)

        # Fiziksel olarak mümkün en büyük gecikme ±90°'de oluşur ve mikrofon
        # aralığına eşittir. Arama penceresini buna bağlamak, uzaktaki sahte
        # tepeleri baştan eler; pay yansımalar ve aralık hatası için.
        max_samples = self.mic_spacing_m / SPEED_OF_SOUND_MPS * self.sample_rate
        self.max_lag = max(2, int(math.ceil(max_samples * 1.5)))

    def estimate(self, left: np.ndarray, right: np.ndarray) -> Tuple[Optional[float], float]:
        """(azimut°, keskinlik). Yön güvenilir değilse azimut None.

        Azimut robot çerçevesinde: 0° ileri, +90° sağ, -90° sol.
        """
        if len(left) != len(right) or len(left) < 64:
            return None, 0.0

        lag, sharpness = _gcc_phat(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            self.max_lag,
        )
        if sharpness < self.min_sharpness:
            return None, sharpness

        path_difference = lag / self.sample_rate * SPEED_OF_SOUND_MPS
        sine = path_difference / self.mic_spacing_m
        if abs(sine) > 1.3:
            # Aralık tahmininin payını da aşıyor: bu gecikme bu çiftten çıkamaz.
            return None, sharpness

        return math.degrees(math.asin(max(-1.0, min(1.0, sine)))), sharpness
