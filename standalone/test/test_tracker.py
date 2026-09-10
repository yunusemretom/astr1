#!/usr/bin/env python3
"""Tests for the ROS-free tracker.

This is the same brain the ROS node runs — the fusion, target manager and FSM are
imported, not reimplemented — so what is tested here is the wiring around them, and
that the guarantees the ROS side already has survive the move: bearings that account
for where the head is, audio that says who rather than where, and a head estimate
that does not collapse to centre when the encoder stays silent.

If behaviour here and under ROS ever differ, the difference is the plumbing. That is
the whole reason this exists.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_path  # noqa: F401
from astro_base.gaze.types import GazeStateEnum, PrioritySource  # noqa: E402

from tracker import Detection, GazeTracker  # noqa: E402


FRAME = (640, 480)


def _face_at(u_fraction: float, confidence: float = 0.92) -> Detection:
    """A face box centred at the given fraction across the frame (0 left, 1 right)."""
    w = h = 90
    cx = u_fraction * FRAME[0]
    return Detection(x=int(cx - w / 2), y=195, w=w, h=h, confidence=confidence)


class TestAFaceMovesTheHead(unittest.TestCase):
    def setUp(self):
        self.tracker = GazeTracker()

    def _settle(self, faces, cycles=40, t0=100.0, head=0.0):
        result = None
        for i in range(cycles):
            result = self.tracker.step(
                faces=faces, frame_size=FRAME, doa_deg=None,
                measured_head_deg=head, timestamp=t0 + i * 0.02,
            )
        return result

    def test_a_face_on_the_left_is_answered_with_a_leftward_command(self):
        """Image left is positive yaw in REP-103, which is the convention the whole
        stack uses and the easiest thing to get backwards."""
        result = self._settle([_face_at(0.15)])

        self.assertGreater(result.target_yaw_deg, 5.0)

    def test_a_face_on_the_right_is_answered_with_a_rightward_command(self):
        result = self._settle([_face_at(0.85)])

        self.assertLess(result.target_yaw_deg, -5.0)

    def test_a_centred_face_leaves_the_head_where_it_is(self):
        result = self._settle([_face_at(0.5)])

        self.assertAlmostEqual(result.target_yaw_deg, 0.0, delta=4.0)

    def test_a_visible_face_owns_the_attention(self):
        result = self._settle([_face_at(0.3)])

        self.assertIn(
            result.owner, (PrioritySource.VISUAL_TRACKING, PrioritySource.ACTIVE_SPEAKER)
        )

    def test_an_empty_scene_stays_idle_at_centre(self):
        result = self._settle([])

        self.assertEqual(result.owner, PrioritySource.IDLE)
        self.assertEqual(result.gaze_state, GazeStateEnum.IDLE)
        self.assertAlmostEqual(result.target_yaw_deg, 0.0, delta=0.1)


class TestBearingsAccountForWhereTheHeadIs(unittest.TestCase):
    """`body_azimuth = head_yaw + camera_azimuth`. Getting this wrong is what made
    the ROS version drive back to centre the moment the head moved."""

    def test_a_centred_face_seen_from_a_turned_head_is_not_at_zero(self):
        tracker = GazeTracker()
        result = None
        for i in range(40):
            result = tracker.step(
                faces=[_face_at(0.5)], frame_size=FRAME, doa_deg=None,
                measured_head_deg=30.0, timestamp=100.0 + i * 0.02,
            )

        self.assertGreater(result.target_yaw_deg, 25.0)


class TestTheHeadEstimateWithoutAnEncoder(unittest.TestCase):
    def test_a_silent_encoder_is_reported_rather_than_assumed_to_be_zero(self):
        tracker = GazeTracker()

        tracker.step(faces=[], frame_size=FRAME, doa_deg=None,
                     measured_head_deg=None, timestamp=100.0)

        self.assertTrue(tracker.head_feedback_missing)

    def test_the_estimate_follows_the_command_instead_of_sitting_at_zero(self):
        tracker = GazeTracker()
        for i in range(400):
            tracker.step(faces=[_face_at(0.05)], frame_size=FRAME, doa_deg=None,
                         measured_head_deg=None, timestamp=100.0 + i * 0.02)

        self.assertGreater(tracker.head_angle_deg, 5.0)

    def test_a_real_reading_takes_over_from_the_estimate(self):
        tracker = GazeTracker()
        for i in range(20):
            tracker.step(faces=[_face_at(0.05)], frame_size=FRAME, doa_deg=None,
                         measured_head_deg=None, timestamp=100.0 + i * 0.02)

        tracker.step(faces=[], frame_size=FRAME, doa_deg=None,
                     measured_head_deg=-12.0, timestamp=101.0)

        self.assertEqual(tracker.head_angle_deg, -12.0)
        self.assertFalse(tracker.head_feedback_missing)


class TestAudioSaysWhoNotWhere(unittest.TestCase):
    """The same guarantee the ROS side has: a wandering DOA must not drag the aim."""

    def test_the_aim_is_unchanged_across_the_doa_spread(self):
        aims = set()
        for raw_doa in (0.0, 15.0, 330.0, 345.0):
            tracker = GazeTracker()
            result = None
            for i in range(40):
                result = tracker.step(
                    faces=[_face_at(0.3)], frame_size=FRAME, doa_deg=raw_doa,
                    measured_head_deg=0.0, timestamp=100.0 + i * 0.02,
                )
            aims.add(round(result.target_yaw_deg, 1))

        self.assertEqual(len(aims), 1, f"audio moved the aim: {aims}")


class TestOnlySpeechEarnsTheHead(unittest.TestCase):
    """Yalnizca insan sesi kafayi cevirebilir.

    Sahadaki 130 saniyelik bir kosuda kafa, yuz kaybolduktan sonra 90 saniye
    boyunca +-75 arasinda salindi: kerteriz ureten her yuksek ses hedefi ele
    geciriyordu ve guven sabit 0.85 verildigi icin 0.45'lik ses edinme esigini
    her seferinde asiyordu. Pencereden gelen bir araba da, bir ugultu da yuksek
    ve israrcidir; onlari enerji de yon kararliligi da elemez.

    Bu iki test ciftin iki yarisi: gurultu kafayi kapamamali, ama konusma hala
    kapabilmeli. Ikincisi olmadan birincisi "sesle takibi kapatmak" olurdu.
    """

    NOISE_BEARING = 60.0

    def _run(self, speech, cycles=60):
        tracker = GazeTracker()
        result = None
        for i in range(cycles):
            result = tracker.step(
                faces=[], frame_size=FRAME, doa_deg=self.NOISE_BEARING,
                speech=speech, measured_head_deg=0.0, timestamp=200.0 + i * 0.02,
            )
        return result

    def test_araba_gurultusu_kafayi_kapamaz(self):
        from astro_audio.speech_detector import SpeechVerdict

        not_speech = SpeechVerdict(is_speech=False, confidence=0.0, harmonicity=0.32,
                                   modulation=0.03, rms=0.2,
                                   reason="ne harmonik ne modulasyonlu")

        result = self._run(not_speech)

        self.assertEqual(result.owner, PrioritySource.IDLE,
                         "konusma olmayan bir ses hedefi ele gecirdi")
        self.assertIsNone(result.target_id)

    def test_konusma_hala_kafayi_kapabilir(self):
        """Gurultuyu elerken sesle takibi de elememis olmaliyiz."""
        from astro_audio.speech_detector import SpeechVerdict

        speech = SpeechVerdict(is_speech=True, confidence=0.76, harmonicity=0.59,
                               modulation=0.83, rms=0.2)

        result = self._run(speech)

        self.assertEqual(result.owner, PrioritySource.ACTIVE_SPEAKER,
                         "konusma kafayi cevirmedi -- gurultu filtresi ozelligi de kapatmis")

    def test_guven_konusma_olcusunden_gelir_sabit_085_ten_degil(self):
        """Sabit 0.85, kestiricinin kendi guveni olcumde ayirt etmedigi icin konmustu
        (dort akustik kosulda 0.40-0.46). Yerine konusma olcusunun guveni geciyor."""
        from astro_audio.speech_detector import SpeechVerdict

        weak = SpeechVerdict(is_speech=True, confidence=0.52, harmonicity=0.47,
                             modulation=0.22, rms=0.2)
        strong = SpeechVerdict(is_speech=True, confidence=0.95, harmonicity=0.90,
                               modulation=0.90, rms=0.2)

        self.assertLess(self._run(weak, cycles=6).confidence,
                        self._run(strong, cycles=6).confidence,
                        "guven konusma olcusunu izlemiyor -- hala sabit")


class TestCalibrationIsRead(unittest.TestCase):
    """Kalibrasyon dosyasi standalone tarafina da ulasmali.

    `GazeTracker` cıplak `CalibrationConfig()` kuruyordu, yani
    astro_base/config/calibration_params.yaml hic okunmuyordu. Onemi su: dizinin
    fiziksel montaj kaymasi olculdugunde yazilacagi yer `audio.yaw_offset_deg` ve
    o deger standalone'a hic ulasmiyordu -- ROS tarafi ayari alir, bu program
    almazdi, ve ikisi ayni beyni calistirdigi iddiasi orada sessizce bozulurdu.
    """

    def _config(self, tmpdir, yaw_offset):
        path = Path(tmpdir) / "calibration_params.yaml"
        path.write_text(
            "/**:\n"
            "  ros__parameters:\n"
            "    audio:\n"
            f"      yaw_offset_deg: {yaw_offset}\n"
            "      invert: true\n",
            encoding="utf-8")
        return path

    def test_dosyadaki_ses_kaymasi_kerterize_uygulanir(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tracker = GazeTracker(calibration_path=self._config(tmp, 25.0))

        self.assertAlmostEqual(tracker.calib.audio.yaw_offset_deg, 25.0)
        self.assertAlmostEqual(
            tracker.transformer.raw_audio_doa_to_head_bearing(0.0), -25.0, places=3,
            msg="kayma okundu ama kerterize uygulanmiyor")

    def test_acikca_verilen_kalibrasyon_dosyayi_ezer(self):
        """Testler ve deneyler icin dosyaya dokunmadan ayar verebilmek gerekiyor."""
        from astro_base.gaze.coordinate_frames import AudioCalibration, CalibrationConfig

        explicit = CalibrationConfig()
        explicit.audio = AudioCalibration(yaw_offset_deg=7.0, invert=True)

        tracker = GazeTracker(calibration=explicit)

        self.assertAlmostEqual(tracker.calib.audio.yaw_offset_deg, 7.0)

    def test_dosya_yoksa_program_varsayilanlarla_calisir(self):
        """Masaustunde depo duzeni farkli olabilir; eksik dosya durdurmamali."""
        tracker = GazeTracker(calibration_path=Path("/yok/boyle/bir/dosya.yaml"))

        self.assertIsNotNone(tracker.calib)
        self.assertAlmostEqual(tracker.calib.head.max_angle_deg, 85.0)


class TestRobotDoesNotChaseItsOwnVoice(unittest.TestCase):
    """Robot konusurken kendi sesi kafayi cevirmemeli.

    `process_raw_doa` `is_robot_speaking` parametresini zaten kabul ediyor ve
    guveni 0.15 ile carpiyor; standalone bu parametreyi hic gecirmiyordu.
    Hoparlor mikrofonun yaninda oldugu icin robot konustugu anda gucla bir
    kerteriz uretilir ve o kerteriz her zaman hoparlorun yonunu gosterir.
    """

    def _run(self, is_robot_speaking):
        from astro_audio.speech_detector import SpeechVerdict

        speech = SpeechVerdict(is_speech=True, confidence=0.85, harmonicity=0.60,
                               modulation=0.85, rms=0.2)
        tracker = GazeTracker()
        result = None
        for i in range(60):
            result = tracker.step(
                faces=[], frame_size=FRAME, doa_deg=55.0, speech=speech,
                is_robot_speaking=is_robot_speaking,
                measured_head_deg=0.0, timestamp=300.0 + i * 0.02,
            )
        return result

    def test_robot_konusurken_ses_hedefi_ele_geciremez(self):
        result = self._run(is_robot_speaking=True)

        self.assertEqual(result.owner, PrioritySource.IDLE,
                         "robot kendi sesine dondu")

    def test_robot_susarken_ayni_ses_hedefi_ele_gecirir(self):
        """Bastirma calisiyor diye ozelligi kapatmis olmayalim."""
        result = self._run(is_robot_speaking=False)

        self.assertEqual(result.owner, PrioritySource.ACTIVE_SPEAKER)


class TestSequentialSpeakersAfterIdle(unittest.TestCase):
    """Sessizlik/bosta kalma sonrasinda farkli yondeki ses kafayi cevirmeli.

    Onceki bir konusmacidan (orn. sagda, -60°) sonra robot bosa (IDLE) dustugunde
    veya zaman asimi gectiginde, soldan gelen yeni bir konusma (orn. +45°)
    'aykiri deger' (outlier) olarak cope atilmamali; kafayi hemen o yone cevirmeli.
    """

    def test_yeni_yondeki_ses_bostayken_hemen_kabul_edilir(self):
        from astro_audio.speech_detector import SpeechVerdict

        speech = SpeechVerdict(
            is_speech=True, confidence=0.80, harmonicity=0.60,
            modulation=0.70, rms=0.2,
        )
        tracker = GazeTracker()

        # 1. Sagdaki konusmaci (doa 60° CW -> -60° REP-103)
        res1 = None
        for i in range(20):
            res1 = tracker.step(
                faces=[], frame_size=FRAME, doa_deg=60.0, speech=speech,
                measured_head_deg=None, timestamp=10.0 + i * 0.05,
            )
        self.assertEqual(res1.owner, PrioritySource.ACTIVE_SPEAKER)
        self.assertLess(res1.target_yaw_deg, -20.0)

        # 2. Sessizlik ve robotun bosa (IDLE) donmesi
        res_idle = None
        for i in range(100):
            t = 15.0 + i * 0.05
            res_idle = tracker.step(
                faces=[], frame_size=FRAME, doa_deg=None, speech=None,
                measured_head_deg=0.0, timestamp=t,
            )
        self.assertEqual(res_idle.gaze_state, GazeStateEnum.IDLE)

        # 3. Soldan yeni konusmaci (doa 300° CW -> +60° REP-103)
        res2 = None
        for i in range(5):
            res2 = tracker.step(
                faces=[], frame_size=FRAME, doa_deg=300.0, speech=speech,
                measured_head_deg=0.0, timestamp=30.0 + i * 0.05,
            )
        self.assertEqual(res2.owner, PrioritySource.ACTIVE_SPEAKER)
        self.assertGreater(res2.target_yaw_deg, 20.0)


if __name__ == "__main__":
    unittest.main()
