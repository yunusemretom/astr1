#!/usr/bin/env python3
"""Konuşma ayırt edici testleri.

Kafayı çevirme yetkisi "bu ses konuşma mı?" sorusuna bağlı. Pencereden gelen bir
araba sesi de, bir cızırtı da ısrarcıdır ve yüksektir; ikisini de enerji ya da
yön kararlılığı elemez. Ayrım sinyalin kendisinde:

  konuşma  — harmonik (F0 ~80-300 Hz) VE hece hızında modülasyonlu (4-8 Hz)
  araba    — harmonik değil, modülasyonsuz (geçerken yavaşça şişer, ~0.5 Hz)
  cızırtı  — harmonik OLABİLİR (şebeke uğultusu 50 Hz + katları) ama modülasyonsuz

Yani tek başına harmoniklik yetmiyor, tek başına modülasyon da yetmiyor; ikisi
birden gerekiyor. Aşağıdaki sahneler bunu kasten zorluyor.

`test_sentetik_sahneler_iddia_edilen_ozellikleri_tasiyor` üretecin kendisini
bağımsız matematikle doğruluyor. Bu depoda sentetik sahne iki kez yanlış soruyu
cevaplattı; sahne doğrulanmadan üstüne test yazılmıyor.
"""

import os
import sys
import unittest

import numpy as np

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

SAMPLE_RATE = 16000


def _normalise(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    return x / peak * 0.5 if peak > 0 else x


def make_speech(duration_s: float = 1.5, f0_hz: float = 120.0, seed: int = 0) -> np.ndarray:
    """Sesli konuşma: harmonik yığın + 5 Hz hece zarfı + duraklar."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE

    # Perde tek düze değil; gerçek konuşmada yavaşça gezinir.
    f0 = f0_hz * (1.0 + 0.06 * np.sin(2 * np.pi * 1.3 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    signal = np.zeros_like(t)
    for harmonic in range(1, 41):
        if harmonic * f0_hz > 4000.0:
            break
        # 1/n düşüş + kabaca formant vurgusu (700 ve 1200 Hz civarı).
        freq = harmonic * f0_hz
        formant = 1.0 + 1.5 * np.exp(-((freq - 700.0) / 250.0) ** 2)
        formant += 1.0 * np.exp(-((freq - 1200.0) / 350.0) ** 2)
        signal += (formant / harmonic) * np.sin(harmonic * phase + rng.uniform(0, 2 * np.pi))

    # Hece zarfı: 5 Hz, aralarda gerçek duraklar.
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5.0 * t - np.pi / 2)
    envelope = envelope ** 2
    signal *= envelope
    signal += rng.normal(0, 0.02 * float(np.std(signal)), len(t))
    return _normalise(signal)


def make_car(duration_s: float = 1.5, seed: int = 1) -> np.ndarray:
    """Geçen araba: alçak frekans ağırlıklı geniş bant, yavaş şişen zarf."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE

    white = rng.normal(0, 1, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    # Motor gürültüsü: ~1/f, 2 kHz üstü hızla söner.
    shape = 1.0 / np.sqrt(np.maximum(freqs, 20.0))
    shape *= np.exp(-freqs / 1500.0)
    signal = np.fft.irfft(spectrum * shape, n)

    # Araba yaklaşıp uzaklaşır: zarf var ama 0.5 Hz, hece hızının çok altında.
    signal *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t - np.pi / 2)
    return _normalise(signal)


def make_buzz(duration_s: float = 1.5, f0_hz: float = 150.0, seed: int = 2) -> np.ndarray:
    """Cızırtı / tonal uğultu: harmonik ama sabit genlikli.

    Bu sahnenin işi harmoniklik ölçüsünü tek başına çürütmek, o yüzden perdesi
    kasten insan konuşma aralığında (150 Hz — bir fan iniltisi, bir anahtarlamalı
    güç kaynağının ötüşü). Otokorelasyon buna "kusursuz harmonik" der. Eksik olan
    tek şey hece modülasyonu: uğultu nefes almaz.

    Perdeyi 50 Hz şebeke uğultusuna koymak sahneyi işlevsiz bırakıyordu — periyodu
    (320 örnek) 80-300 Hz perde arama aralığının dışında kalıyor ve harmoniklik
    ölçüsü onu zaten görmüyor. Ölçülmüş: pitch_strength 0.17.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    signal = np.zeros_like(t)
    for harmonic in range(1, 40):
        freq = f0_hz * harmonic
        if freq > 6000.0:
            break
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    signal += rng.normal(0, 0.01 * float(np.std(signal)), len(t))
    return _normalise(signal)


# --- Sahneyi bağımsız olarak ölçen yardımcılar (test edilen modülden ayrı) ---

def _envelope_modulation_ratio(x: np.ndarray, low_hz: float, high_hz: float) -> float:
    """Zarfin [low, high] bandindaki enerjisinin toplam AC enerjisine orani."""
    envelope = np.abs(x)
    # Kaba alcak geciren: 100 Hz'e indir, zarf zaten yavas.
    step = SAMPLE_RATE // 200
    envelope = np.array([envelope[i:i + step].mean() for i in range(0, len(envelope) - step, step)])
    envelope = envelope - envelope.mean()
    if len(envelope) < 16 or not np.any(envelope):
        return 0.0
    fs_env = 200.0
    spectrum = np.abs(np.fft.rfft(envelope)) ** 2
    freqs = np.fft.rfftfreq(len(envelope), 1.0 / fs_env)
    band = spectrum[(freqs >= low_hz) & (freqs <= high_hz)].sum()
    total = spectrum.sum()
    return float(band / total) if total > 0 else 0.0


def _autocorrelation_pitch_strength(x: np.ndarray) -> float:
    """80-300 Hz perde araliginda normalize otokorelasyon tepesi."""
    x = x - x.mean()
    if not np.any(x):
        return 0.0
    correlation = np.correlate(x, x, mode="full")[len(x) - 1:]
    correlation /= correlation[0]
    lo = int(SAMPLE_RATE / 300.0)
    hi = int(SAMPLE_RATE / 80.0)
    return float(np.max(correlation[lo:hi]))


class TestSyntheticScenes(unittest.TestCase):
    """Once sahnenin kendisi dogru mu? Uretec yanlissa ustundeki test yalan soyler."""

    def test_sentetik_sahneler_iddia_edilen_ozellikleri_tasiyor(self):
        speech, car, buzz = make_speech(), make_car(), make_buzz()

        # 1. Konusma hem harmonik hem hece hizinda modulasyonlu.
        self.assertGreater(_autocorrelation_pitch_strength(speech), 0.5,
                           "konusma sahnesi harmonik degil")
        self.assertGreater(_envelope_modulation_ratio(speech, 4.0, 8.0), 0.25,
                           "konusma sahnesinde hece modulasyonu yok")

        # 2. Araba harmonik degil ve hece hizinda modulasyonsuz.
        self.assertLess(_autocorrelation_pitch_strength(car), 0.5,
                        "araba sahnesi istemeden harmonik cikti")
        self.assertLess(_envelope_modulation_ratio(car, 4.0, 8.0), 0.25,
                        "araba sahnesi istemeden hece hizinda modulasyonlu cikti")

        # 3. Cizirti harmonik AMA modulasyonsuz -- harmonikligi tek basina curuten sahne.
        self.assertGreater(_autocorrelation_pitch_strength(buzz), 0.5,
                           "cizirti sahnesi harmonik degil; harmoniklik testini zorlamiyor")
        self.assertLess(_envelope_modulation_ratio(buzz, 4.0, 8.0), 0.25,
                        "cizirti sahnesi istemeden modulasyonlu cikti")


class TestSpeechDetector(unittest.TestCase):
    """Kafayi cevirme yetkisi burada veriliyor: konusma mi, degil mi."""

    def setUp(self):
        from astro_audio.speech_detector import SpeechDetector

        self.detector = SpeechDetector(sample_rate=SAMPLE_RATE)

    def test_sesli_konusma_konusma_olarak_siniflaniyor(self):
        verdict = self.detector.classify(make_speech())

        self.assertTrue(verdict.is_speech)

    def test_araba_sesi_konusma_sayilmiyor(self):
        """Pencereden gelen trafik yuksek ve israrcidir; kafayi cevirmemeli."""
        verdict = self.detector.classify(make_car())

        self.assertFalse(verdict.is_speech)

    def test_tonal_cizirti_harmonik_olmasina_ragmen_konusma_sayilmiyor(self):
        """Harmoniklik tek basina yetmez: ugultu kusursuz harmoniktir ama nefes almaz.

        Harmoniklige bakip modulasyona bakmayan bir uygulama bu testte duser.
        """
        verdict = self.detector.classify(make_buzz())

        self.assertFalse(verdict.is_speech)


if __name__ == "__main__":
    unittest.main()
