#!/usr/bin/env python3
"""İki mikrofonlu yön bulma testleri.

Bu yol, elde 4'lü dizi yokken var: laptopun dahili çifti gerçekten uzamsal olarak
ayrı ve tek eksende yön veriyor. Burada korunan asıl şey **işaret**. Ölçek yanlışsa
kafa az ya da çok döner, düzeltilebilir bir hata; işaret yanlışsa kafa sesin tam
tersine döner ve bu, hiç dönmemekten kötüdür.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_path  # noqa: F401
from astro_audio.speech_detector import SpeechVerdict  # noqa: E402
from astro_base.gaze.types import PrioritySource  # noqa: E402
from sources import AudioSource  # noqa: E402
from tracker import Detection, GazeTracker  # noqa: E402
from stereo_doa import (  # noqa: E402
    DEFAULT_MIC_SPACING_M,
    SPEED_OF_SOUND_MPS,
    StereoDOA,
)

FS = 44100
SAMPLES = 8192


def _pair(azimuth_deg, spacing_m=DEFAULT_MIC_SPACING_M, seed=4):
    """Belirli bir yönden gelen sesin iki mikrofonda oluşturduğu çift.

    Sağdan gelen ses sağ mikrofona önce ulaşır, yani sol kanal geriden gelir.
    """
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(SAMPLES)
    lag = math.sin(math.radians(azimuth_deg)) * spacing_m / SPEED_OF_SOUND_MPS * FS

    freqs = np.fft.rfftfreq(SAMPLES)
    delayed = np.fft.irfft(
        np.fft.rfft(signal) * np.exp(-2j * np.pi * freqs * lag), SAMPLES)
    return delayed, signal


class StereoBearingTests(unittest.TestCase):
    def setUp(self):
        self.est = StereoDOA(sample_rate=FS)

    def test_sound_on_the_right_reads_positive(self):
        """İşaret sözleşmesi: +90° sağ, -90° sol — 4'lü kestiricininkiyle aynı.

        Aynı olması şart, çünkü aşağıdaki `invert` katmanı bu sözleşmeye göre yazılmış.
        """
        azimuth, _ = self.est.estimate(*_pair(45.0))
        self.assertIsNotNone(azimuth)
        self.assertGreater(azimuth, 0.0)

    def test_sound_on_the_left_reads_negative(self):
        azimuth, _ = self.est.estimate(*_pair(-45.0))
        self.assertIsNotNone(azimuth)
        self.assertLess(azimuth, 0.0)

    def test_angles_are_recovered_across_the_frontal_arc(self):
        for truth in (-80.0, -45.0, -20.0, 0.0, 20.0, 45.0, 80.0):
            with self.subTest(truth=truth):
                azimuth, _ = self.est.estimate(*_pair(truth))
                self.assertIsNotNone(azimuth)
                self.assertLess(abs(azimuth - truth), 8.0)

    def test_incoherent_channels_yield_nothing(self):
        """Oda gürültüsü keskin bir tepe vermez; yön iddia edilmemeli."""
        rng = np.random.default_rng(11)
        azimuth, sharpness = self.est.estimate(
            rng.standard_normal(SAMPLES), rng.standard_normal(SAMPLES))

        self.assertIsNone(azimuth)
        self.assertLess(sharpness, self.est.min_sharpness)

    def test_a_delay_too_large_for_the_pair_is_refused(self):
        """Bu aralıktan fiziksel olarak çıkamayacak gecikme, yansıma ya da hatadır."""
        wide = StereoDOA(sample_rate=FS, mic_spacing_m=0.30)
        azimuth, _ = wide.estimate(*_pair(60.0, spacing_m=0.30))
        self.assertIsNotNone(azimuth, "kendi aralığıyla üretilen gecikme kabul edilmeli")

        narrow = StereoDOA(sample_rate=FS, mic_spacing_m=0.005)
        far_azimuth, _ = narrow.estimate(*_pair(60.0, spacing_m=0.30))
        self.assertIsNone(far_azimuth)

    def test_the_search_window_follows_the_spacing(self):
        """Geniş aralık daha büyük gecikme görebilmeli, dar aralık görmemeli."""
        self.assertGreater(StereoDOA(sample_rate=FS, mic_spacing_m=0.20).max_lag,
                           StereoDOA(sample_rate=FS, mic_spacing_m=0.04).max_lag)

    def test_sub_sample_resolution_is_actually_used(self):
        """44.1 kHz'de tam örnek adımı ~8°: yuvarlarsak yön kullanılamaz kabalıkta olur."""
        seen = {self.est.estimate(*_pair(t))[0] for t in (10.0, 12.0, 14.0)}
        self.assertEqual(len(seen), 3, "farklı açılar aynı değere yuvarlanıyor")


class BearingReachesTheHeadUnturnedTests(unittest.TestCase):
    """Mikrofondan kafa komutuna kadar tüm zincir, tek iddiada: yön aynı olmalı.

    Zincirde iki sözleşme var. `StereoDOA` 4'lü kestiricinin sözleşmesinde üretiyor
    (0 ön, +90 sağ), `AudioSource` negatifi +360'a sarıyor, ve aşağıdaki `invert`
    katmanı bunu REP-103 gövde açısına çeviriyor (pozitif = sol). Üçü de doğru ama
    biri değişirse kafa sesin tersine döner ve hiçbir birim testi bunu görmez —
    hepsi kendi içinde tutarlı kalır. Bu yüzden buradaki ölçüt mutlak açı değil,
    sesin ve gözün aynı tarafa komut vermesi.
    """

    FRAME = (640, 480)

    def _command_for(self, faces, doa_deg, steps=12):
        """Bir kerteriz verildiginde konusma verdisi de verilir.

        Kafayi yalnizca insan sesi cevirebiliyor: kerteriz uretmek yetmiyor, o
        kerterizin geldigi pencerenin konusma olarak siniflanmasi da gerekiyor
        (bkz. astro_audio.speech_detector). Buradaki iddia isaret sozlesmesi
        hakkinda, siniflandirma hakkinda degil -- o yuzden sahne konusmayla
        besleniyor. Verdi verilmezse ses hic islenmez ve `by_ear` sifirda kalir;
        soldaki es test tam bu yuzden bir sure yanlis sebeple geciyordu:
        copysign(1.0, 0.0) yuz tarafinin isaretiyle rastlantisal olarak esitti.
        """
        speech = SpeechVerdict(is_speech=True, confidence=0.80, harmonicity=0.60,
                               modulation=0.80, rms=0.2) if doa_deg is not None else None
        tracker = GazeTracker()
        result = None
        for i in range(steps):
            result = tracker.step(faces=faces, frame_size=self.FRAME, doa_deg=doa_deg,
                                  speech=speech,
                                  measured_head_deg=0.0, timestamp=1.0 + i * 0.1)
        return result

    def _bearing_from_a_pair(self, azimuth_deg):
        """Sentetik stereo çiftini AudioSource'un yayınladığı kerterize kadar götürür."""
        source = AudioSource()
        source._stereo = StereoDOA(sample_rate=FS)
        left, right = _pair(azimuth_deg)
        source.process_stereo_block(np.stack([left, right], axis=1), timestamp=1.0)
        return source.latest_doa_deg(1.0)

    def test_a_pair_on_the_right_drives_the_head_the_way_a_face_on_the_right_does(self):
        bearing = self._bearing_from_a_pair(45.0)
        self.assertIsNotNone(bearing, "sağdan gelen ses kerteriz üretmedi")

        by_ear = self._command_for([], bearing)
        by_eye = self._command_for(
            [Detection(x=520, y=200, w=80, h=80, confidence=0.95)], None)

        self.assertEqual(math.copysign(1.0, by_ear.target_yaw_deg),
                         math.copysign(1.0, by_eye.target_yaw_deg),
                         "ses ve görüntü kafayı zıt yönlere çeviriyor")

    def test_the_same_holds_on_the_left(self):
        bearing = self._bearing_from_a_pair(-45.0)
        self.assertIsNotNone(bearing)

        by_ear = self._command_for([], bearing)
        by_eye = self._command_for(
            [Detection(x=40, y=200, w=80, h=80, confidence=0.95)], None)

        self.assertEqual(math.copysign(1.0, by_ear.target_yaw_deg),
                         math.copysign(1.0, by_eye.target_yaw_deg))

    def test_a_voice_with_no_face_is_what_takes_over(self):
        """İstenen davranış buydu: ekranda yüz yokken seslenene dönmek."""
        result = self._command_for([], self._bearing_from_a_pair(40.0))

        self.assertEqual(result.owner, PrioritySource.ACTIVE_SPEAKER)
        self.assertIsNotNone(result.target_id)


if __name__ == "__main__":
    unittest.main()
