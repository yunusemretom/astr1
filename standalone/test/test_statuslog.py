#!/usr/bin/env python3
"""Durum akışı testleri.

Bu log ekransız koşuda tek görünürlük kaynağı, o yüzden burada korunan iki şey var:
bir durum geçişinin nabız aralığına takılıp kaybolmaması, ve satırın komut ile ölçülen
açı arasındaki farkı göstermesi — README'deki teşhis tablosu ("istenen değişiyor,
gercek sabit -> aktüatör") o farkı okumaya dayanıyor.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_path  # noqa: F401
from statuslog import StatusLog  # noqa: E402


class _Result:
    """GazeResult'in loga giren alanları."""

    def __init__(self, state="TRACKING", owner="VISUAL_TRACKING", target="person_1",
                 conf=0.90, wanted=10.0, actual=8.0):
        self.gaze_state = state
        self.owner = owner
        self.target_id = target
        self.confidence = conf
        self.target_yaw_deg = wanted
        self.head_angle_deg = actual


class StatusLogTests(unittest.TestCase):
    def setUp(self):
        self.out = []
        self.log = StatusLog(interval_s=1.0, printer=self.out.append)

    def test_first_update_always_prints(self):
        self.assertIsNotNone(self.log.update(0.0, _Result()))
        self.assertEqual(len(self.out), 1)

    def test_quiet_between_beats(self):
        self.log.update(0.0, _Result())
        self.assertIsNone(self.log.update(0.3, _Result()))
        self.assertIsNone(self.log.update(0.9, _Result()))
        self.assertEqual(len(self.out), 1)

    def test_beat_after_the_interval(self):
        self.log.update(0.0, _Result())
        self.assertIsNotNone(self.log.update(1.1, _Result()))
        self.assertEqual(len(self.out), 2)

    def test_state_change_does_not_wait_for_the_next_beat(self):
        """A transition inside the beat interval is the thing worth seeing.

        Losing one heartbeat costs nothing; losing the moment the head decided to turn
        costs the reason the log exists.
        """
        self.log.update(0.0, _Result(state="TRACKING"))
        line = self.log.update(0.1, _Result(state="ORIENTING"))

        self.assertIsNotNone(line)
        self.assertIn("ORIENTING", line)
        self.assertTrue(line.startswith(">"), "change lines should be marked")

    def test_owner_and_target_changes_also_break_through(self):
        self.log.update(0.0, _Result(owner="IDLE", target=None))
        self.assertIsNotNone(self.log.update(0.1, _Result(owner="ACTIVE_SPEAKER", target=None)))
        self.assertIsNotNone(self.log.update(0.2, _Result(owner="ACTIVE_SPEAKER", target="audio_speaker_1")))

    def test_line_shows_command_actual_and_their_difference(self):
        """The gap between commanded and measured is how the README's table is read."""
        line = self.log.update(0.0, _Result(wanted=20.0, actual=5.0), head_feedback=True)

        self.assertIn("istenen +20.0", line)
        self.assertIn("gercek  +5.0", line)
        self.assertIn("fark +15.0", line)

    def test_missing_audio_bearing_reads_as_absent_not_zero(self):
        """A silent microphone must not look like a sound coming from dead ahead."""
        line = self.log.update(0.0, _Result(), doa_deg=None)

        self.assertNotIn("ses  +0.0", line)
        self.assertIn("ses", line)

    def test_audio_bearing_is_shown_when_present(self):
        line = self.log.update(0.0, _Result(), doa_deg=-40.0)
        self.assertIn("-40.0", line)

    def test_zero_interval_prints_only_on_change(self):
        log = StatusLog(interval_s=0.0, printer=self.out.append)
        log.update(0.0, _Result(state="TRACKING"))
        self.assertIsNone(log.update(5.0, _Result(state="TRACKING")))
        self.assertIsNotNone(log.update(6.0, _Result(state="ORIENTING")))

    def test_head_feedback_flag_is_reported(self):
        with_fb = self.log.update(0.0, _Result(), head_feedback=True)
        self.assertIn("kafa:V", with_fb)

        log2 = StatusLog(interval_s=1.0, printer=self.out.append)
        without = log2.update(0.0, _Result(), head_feedback=False)
        self.assertIn("kafa:X", without)

    def test_encodersiz_aci_olcum_diye_sunulmaz(self):
        line = self.log.update(0.0, _Result(actual=8.0), head_feedback=False)
        self.assertIn("tahmin", line)
        self.assertNotIn("gercek", line)

    def test_summary_counts_what_was_written(self):
        self.log.update(0.0, _Result())
        self.log.update(1.1, _Result())

        summary = self.log.summary(elapsed_s=2.0, frames=60)
        self.assertIn("60 kare", summary)
        self.assertIn("30.0 Hz", summary)
        self.assertIn("2 durum", summary)




class SpeechInTheStatusLineTests(unittest.TestCase):
    """Kafanin neden donmedigi satirdan okunabilmeli.

    Kafayi artik yalnizca konusma cevirebiliyor. Bu, ekransiz kosuda yeni bir
    sessiz basarisizlik yolu aciyor: kerteriz uretiliyor, kafa donmuyor, ve
    sebebi hicbir yerde yazmiyor. README'deki teshis tablosunun mantigi burada
    da gecerli -- hangi katmanin sustugu tahmin edilmemeli, okunmali.
    """

    def setUp(self):
        self.out = []
        self.log = StatusLog(interval_s=1.0, printer=self.out.append)

    def test_konusma_reddedildiginde_sebep_satirda_gorunur(self):
        class _Verdict:
            is_speech = False
            confidence = 0.0
            reason = "harmonik ama hece modulasyonu yok"

        self.log.update(0.0, _Result(), doa_deg=90.0, speech=_Verdict())

        self.assertIn("hece", self.out[0],
                      f"kerteriz elendi ama sebebi satirda yok: {self.out[0]}")

    def test_konusma_kabul_edildiginde_satir_bunu_soyler(self):
        class _Verdict:
            is_speech = True
            confidence = 0.76
            reason = ""

        self.log.update(0.0, _Result(), doa_deg=90.0, speech=_Verdict())

        self.assertIn("konusma", self.out[0].lower(),
                      f"kabul edilen konusma satirda gorunmuyor: {self.out[0]}")

    def test_yon_olmasa_da_konusma_olculeri_gorunur(self):
        from astro_audio.speech_detector import SpeechVerdict

        speech = SpeechVerdict(False, 0.0, 0.71, 0.08, 0.01234,
                               "harmonik ama hece modulasyonu yok")
        line = self.log.update(0.0, _Result(), doa_deg=None, speech=speech)

        self.assertIn("hece modulasyonu yok", line)
        self.assertIn("rms=0.01234", line)
        self.assertIn("harm=0.710", line)
        self.assertIn("mod=0.080", line)


if __name__ == "__main__":
    unittest.main()
