#!/usr/bin/env python3
"""Comprehensive Unit Tests for ReSpeaker 4-Mic Acoustic DOA Estimator and Controlled Left/Right Validation."""

import math
import os
import sys
import unittest
import numpy as np

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)

from astro_audio.doa_estimator import AcousticDOAEstimator, ReSpeakerGeometry, gcc_phat


def generate_multichannel_synthetic_sound(
    azimuth_deg: float,
    duration_s: float = 0.05,
    sample_rate: int = 16000,
    seed: int = 42,
) -> np.ndarray:
    """Generates broadband 4-channel audio representing a sound source at given azimuth.
    
    Geometry:
      Mic 0 (Front): (0.0, +R)
      Mic 1 (Right): (+R, 0.0)
      Mic 2 (Back):  (0.0, -R)
      Mic 3 (Left):  (-R, 0.0)
    """
    n_samples = int(duration_s * sample_rate)
    rad = math.radians(azimuth_deg)
    src_dir = np.array([math.sin(rad), math.cos(rad)])
    
    r = ReSpeakerGeometry.RADIUS_M
    mic_pos = np.array([
        [0.0, +r],   # Mic 0: Front
        [+r, 0.0],   # Mic 1: Right
        [0.0, -r],   # Mic 2: Back
        [-r, 0.0],   # Mic 3: Left
    ])
    
    c = ReSpeakerGeometry.SPEED_OF_SOUND_MPS
    delays = -np.dot(mic_pos, src_dir) / c
    delays -= np.min(delays)
    
    np.random.seed(seed)
    base_signal = np.random.normal(0, 1, n_samples * 2)
    channels = np.zeros((4, n_samples), dtype=np.float32)

    # Gecikme ILERI degil GERI kaydirir. `delays` normalize edildikten sonra
    # kaynaga en uzak mikrofonun degeri en buyuktur, ve o mikrofon sesi en GEC
    # duymalidir; base dizisinden daha ERKEN bir indeksten ornekleyerek olur.
    #
    # Bu satir uzun sure `+ shift_samples` idi, yani uzak mikrofon once duyuyordu.
    # Kestiricideki 180 derecelik isaret hatasiyla birlikte iki yanlis birbirini
    # goturuyor ve asagidaki testlerin hepsi geciyordu; sahne, olcmesi gereken
    # hatayi uretiyordu. Ikisi ayni anda duzeltildi.
    origin = n_samples // 2
    for i in range(4):
        shift_samples = delays[i] * sample_rate
        idx = np.arange(n_samples) + origin - shift_samples
        channels[i] = np.interp(idx, np.arange(len(base_signal)), base_signal) * 10000.0

    return channels


def shifted_channels(right_lead_samples: int, front_lead_samples: int = 0,
                     length: int = 2048, seed: int = 3) -> np.ndarray:
    """Dort kanal, tamsayi ornek kaydirmasiyla ve acikca yazilmis fizikle.

    Yukaridaki `generate_multichannel_synthetic_sound` uretecine kasten
    guvenilmiyor. O uretec gecikmeyi `np.interp(arange(n) + shift, ...)` ile
    uyguluyor: ileri indeksten ornekleme sinyali ONE alir, yani "gecikme" dedigi
    sey aslinda erkenlik. Kestiricideki isaret hatasiyla birlikte iki yanlis
    birbirini goturdugu icin o testler gecti ve hata yillarca gorunmedi.

    Burada tek bir kural var ve tartismaya kapali: bir kaynaga daha yakin olan
    mikrofon sesi daha ERKEN duyar. Erkenlik, base dizisinden daha ILERI bir
    indeksten baslamak demektir.

        right_lead_samples > 0  -> kaynak sagda (sag mikrofon once duyar)
        front_lead_samples > 0  -> kaynak onde  (on mikrofon once duyar)

    Kanal sirasi kestiricinin sozlesmesi: 0 on, 1 sag, 2 arka, 3 sol.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, length * 3) * 5000.0
    origin = length

    def take(lead: int) -> np.ndarray:
        return base[origin + lead: origin + lead + length]

    return np.stack([
        take(+front_lead_samples),   # 0 on
        take(+right_lead_samples),   # 1 sag
        take(-front_lead_samples),   # 2 arka
        take(-right_lead_samples),   # 3 sol
    ])


class TestTheAngleIsNotMirrored(unittest.TestCase):
    """Sagdan gelen ses saga, soldan gelen sola raporlanmali.

    Bu, sahadaki 130 saniyelik kosuda kafanin konusanin TERSINE donmesinin
    kaynagi. `gcc_phat`'in docstring'i "tau: positive if refsig lags sig" diyor,
    ama uygulamasi bunun tam tersini donduruyor -- olculdu: refsig sig'den 3
    ornek geride oldugunda tau -3.00 cikiyor. Asagidaki `delta_x = -tau * c`
    satirlari ise *docstring'e* gore yazilmis. Sonuc 180 derecelik sabit hata.

    Donanim gerekmiyor: bunlarin hicbiri ReSpeaker'in kanal sirasina bagli degil,
    yalnizca hangi mikrofonun once duydugunu bilmeye bagli.
    """

    def setUp(self):
        self.estimator = AcousticDOAEstimator(sample_rate=16000)

    def test_sag_mikrofon_once_duyunca_azimut_pozitif(self):
        azimuth, _conf, valid = self.estimator.estimate_from_multichannel_pcm(
            shifted_channels(right_lead_samples=4))

        self.assertTrue(valid)
        self.assertAlmostEqual(azimuth, 90.0, delta=10.0,
                               msg="sagdan gelen ses sol olarak raporlandi")

    def test_sol_mikrofon_once_duyunca_azimut_negatif(self):
        azimuth, _conf, valid = self.estimator.estimate_from_multichannel_pcm(
            shifted_channels(right_lead_samples=-4))

        self.assertTrue(valid)
        self.assertAlmostEqual(azimuth, -90.0, delta=10.0,
                               msg="soldan gelen ses sag olarak raporlandi")

    def test_on_mikrofon_once_duyunca_azimut_sifira_yakin(self):
        azimuth, _conf, valid = self.estimator.estimate_from_multichannel_pcm(
            shifted_channels(right_lead_samples=0, front_lead_samples=4))

        self.assertTrue(valid)
        self.assertAlmostEqual(azimuth, 0.0, delta=10.0,
                               msg="onden gelen ses arkadan geliyor olarak raporlandi")

    def test_gcc_phat_docstringinde_yazan_isareti_dondurur(self):
        """refsig, sig'den geride ise tau pozitif olmali -- fonksiyonun kendi sozu."""
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 4096)
        sig = base[100:1100]
        refsig = base[97:1097]      # refsig, sig'den 3 ornek GERIDE

        tau, _quality = gcc_phat(sig, refsig, fs=16000,
                                 max_tau=ReSpeakerGeometry.PAIR_DIST_M / 343.0)

        self.assertAlmostEqual(tau * 16000, 3.0, delta=0.5,
                               msg="gcc_phat docstring'inin tersi isareti donduruyor")


class TestAcousticDOAEstimator(unittest.TestCase):
    """Tests GCC-PHAT and spatial triangulation across left/right/front/back directions."""

    def setUp(self):
        self.estimator = AcousticDOAEstimator(sample_rate=16000, min_energy_threshold=300.0)

    def test_controlled_right_sound_source(self):
        """Right (+90°) sound source should produce a positive azimuth (+90° ± 5°)."""
        pcm = generate_multichannel_synthetic_sound(azimuth_deg=90.0)
        azimuth, conf, valid = self.estimator.estimate_from_multichannel_pcm(pcm)
        
        self.assertTrue(valid)
        self.assertIsNotNone(azimuth)
        self.assertGreater(azimuth, 0.0)
        self.assertAlmostEqual(azimuth, 90.0, delta=5.0)
        self.assertGreaterEqual(conf, 0.40)

    def test_controlled_left_sound_source(self):
        """Left (-90°) sound source should produce a negative azimuth (-90° ± 5°)."""
        pcm = generate_multichannel_synthetic_sound(azimuth_deg=-90.0)
        azimuth, conf, valid = self.estimator.estimate_from_multichannel_pcm(pcm)
        
        self.assertTrue(valid)
        self.assertIsNotNone(azimuth)
        self.assertLess(azimuth, 0.0)
        self.assertAlmostEqual(azimuth, -90.0, delta=5.0)
        self.assertGreaterEqual(conf, 0.40)

    def test_controlled_front_right_sound_source(self):
        """Front-Right (+45°) sound source should produce positive azimuth (+45° ± 5°)."""
        pcm = generate_multichannel_synthetic_sound(azimuth_deg=45.0)
        azimuth, conf, valid = self.estimator.estimate_from_multichannel_pcm(pcm)
        
        self.assertTrue(valid)
        self.assertIsNotNone(azimuth)
        self.assertAlmostEqual(azimuth, 45.0, delta=5.0)

    def test_controlled_front_left_sound_source(self):
        """Front-Left (-45°) sound source should produce negative azimuth (-45° ± 5°)."""
        pcm = generate_multichannel_synthetic_sound(azimuth_deg=-45.0)
        azimuth, conf, valid = self.estimator.estimate_from_multichannel_pcm(pcm)
        
        self.assertTrue(valid)
        self.assertIsNotNone(azimuth)
        self.assertAlmostEqual(azimuth, -45.0, delta=5.0)

    def test_controlled_front_sound_source(self):
        """Front (0°) sound source should produce azimuth ~0° with high confidence."""
        pcm = generate_multichannel_synthetic_sound(azimuth_deg=0.0)
        azimuth, conf, valid = self.estimator.estimate_from_multichannel_pcm(pcm)
        
        self.assertTrue(valid)
        self.assertIsNotNone(azimuth)
        self.assertAlmostEqual(azimuth, 0.0, delta=5.0)

    def test_spatial_triangulation_multi_angle_series(self):
        """Validates that as sound location shifts (SOL -> ÖN -> SAĞ), estimated azimuth changes accordingly."""
        # 1. Left series (expect negative angles)
        left_angles = [-30.0, -42.0, -38.0]
        left_results = []
        for ang in left_angles:
            pcm = generate_multichannel_synthetic_sound(azimuth_deg=ang, seed=int(abs(ang) * 10))
            az, _, valid = self.estimator.estimate_from_multichannel_pcm(pcm)
            self.assertTrue(valid)
            self.assertLess(az, 0.0)
            self.assertAlmostEqual(az, ang, delta=5.0)
            left_results.append(az)

        # 2. Front series (expect near zero angles)
        front_angles = [-4.0, +2.0, +5.0]
        front_results = []
        for ang in front_angles:
            pcm = generate_multichannel_synthetic_sound(azimuth_deg=ang, seed=int(abs(ang) * 10 + 100))
            az, _, valid = self.estimator.estimate_from_multichannel_pcm(pcm)
            self.assertTrue(valid)
            self.assertAlmostEqual(az, ang, delta=5.0)
            front_results.append(az)

        # 3. Right series (expect positive angles)
        right_angles = [+35.0, +44.0, +39.0]
        right_results = []
        for ang in right_angles:
            pcm = generate_multichannel_synthetic_sound(azimuth_deg=ang, seed=int(abs(ang) * 10 + 200))
            az, _, valid = self.estimator.estimate_from_multichannel_pcm(pcm)
            self.assertTrue(valid)
            self.assertGreater(az, 0.0)
            self.assertAlmostEqual(az, ang, delta=5.0)
            right_results.append(az)

        # Verify distinct positive, near-zero, and negative separation
        self.assertTrue(all(l < -20.0 for l in left_results))
        self.assertTrue(all(-10.0 <= f <= 10.0 for f in front_results))
        self.assertTrue(all(r > 20.0 for r in right_results))


if __name__ == "__main__":
    unittest.main()
