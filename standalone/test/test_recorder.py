#!/usr/bin/env python3
"""Kaydedici testleri.

Kayıt teşhis aracıdır: takılmayı görmek için var. Bu yüzden burada asıl korunan iki
şey, kaydın kare hızı konusunda dürüst olması ve yazıcı açılamadığında takibi
durdurmaması. İkisi de donanımsız test edilir — gerçek bir VideoWriter yerine sahte
bir yazıcı verilir.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_path  # noqa: F401
from recorder import OverlayRecorder, default_path  # noqa: E402


def _frame(width=64, height=48):
    return np.zeros((height, width, 3), dtype=np.uint8)


class _FakeWriter:
    """cv2.VideoWriter yerine geçer; ne aldığını sayar."""

    def __init__(self, path, fourcc, fps, size, opened=True):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False
        self._opened = opened

    def isOpened(self):
        return self._opened

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class _Factory:
    """Üretilen sahte yazıcıyı testin görebilmesi için saklar."""

    def __init__(self, opened=True, raises=None):
        self.opened = opened
        self.raises = raises
        self.made = None

    def __call__(self, path, fourcc, fps, size):
        if self.raises:
            raise self.raises
        self.made = _FakeWriter(path, fourcc, fps, size, opened=self.opened)
        return self.made


class RecorderTests(unittest.TestCase):
    def test_writes_the_frames_it_was_given(self):
        factory = _Factory()
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, measure_seconds=0.1, writer_factory=factory)

        for i in range(5):
            rec.add(_frame(), timestamp=i * 0.1)
        rec.close()

        self.assertEqual(factory.made.frames.__len__(), 5)
        self.assertEqual(rec.frames_written, 5)
        self.assertTrue(factory.made.released)

    def test_buffered_frames_survive_the_measurement_window(self):
        """Ölçüm sırasında gelen kareler atılmaz, dosya açılınca yazılır."""
        factory = _Factory()
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, measure_seconds=1.0, writer_factory=factory)

        # Ölçüm penceresi dolmadan üç kare: yazıcı henüz yok.
        for i in range(3):
            rec.add(_frame(), timestamp=i * 0.1)
        self.assertIsNone(factory.made)

        # Pencere dolunca dosya açılır ve önceki kareler de yazılır.
        rec.add(_frame(), timestamp=1.2)
        self.assertIsNotNone(factory.made)
        self.assertEqual(rec.frames_written, 4)

    def test_frame_rate_matches_what_actually_happened(self):
        """10 Hz koşan bir döngü 10 fps yazmalı — 30 değil.

        Yanlış hız kaydı zamanlama konusunda yalancı yapar: teşhis edilmek istenen
        takılma, hızlı oynatmada kaybolur.
        """
        factory = _Factory()
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, measure_seconds=0.5, writer_factory=factory)

        for i in range(7):  # 0.0 .. 0.6 sn arası, 0.1 sn aralıklarla
            rec.add(_frame(), timestamp=i * 0.1)
        rec.close()

        self.assertAlmostEqual(rec.fps, 10.0, delta=0.6)

    def test_slow_loop_is_recorded_as_slow(self):
        """3 Hz'lik takılan bir döngü 3 fps olarak kaydedilir."""
        factory = _Factory()
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, measure_seconds=0.9, writer_factory=factory)

        for i in range(4):
            rec.add(_frame(), timestamp=i / 3.0)
        rec.close()

        self.assertAlmostEqual(rec.fps, 3.0, delta=0.5)

    def test_unopenable_writer_disables_recording_without_raising(self):
        """Codec yoksa program çökmez; kaydedici kendini kapatır ve sebebini söyler."""
        factory = _Factory(opened=False)
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, measure_seconds=0.1, writer_factory=factory)

        rec.add(_frame(), timestamp=0.0)
        rec.add(_frame(), timestamp=0.2)

        self.assertFalse(rec.active)
        self.assertIsNotNone(rec.error)
        self.assertIn("Kayıt yapılamadı", rec.close())

    def test_writer_exception_is_contained(self):
        """Yazıcı kurucusu patlarsa da takip devam edebilmeli."""
        factory = _Factory(raises=RuntimeError("codec yok"))
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, measure_seconds=0.1, writer_factory=factory)

        rec.add(_frame(), timestamp=0.0)
        rec.add(_frame(), timestamp=0.2)

        self.assertFalse(rec.active)
        self.assertIn("codec yok", rec.error)

    def test_frames_after_failure_are_ignored(self):
        factory = _Factory(opened=False)
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, measure_seconds=0.1, writer_factory=factory)

        rec.add(_frame(), timestamp=0.0)
        rec.add(_frame(), timestamp=0.2)
        rec.add(_frame(), timestamp=0.4)

        self.assertEqual(rec.frames_written, 0)

    def test_early_exit_still_writes_what_was_buffered(self):
        """Ölçüm penceresi dolmadan çıkılsa bile kareler kaybolmaz."""
        factory = _Factory()
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, measure_seconds=10.0, writer_factory=factory)

        rec.add(_frame(), timestamp=0.0)
        rec.add(_frame(), timestamp=0.1)
        summary = rec.close()

        self.assertEqual(rec.frames_written, 2)
        self.assertIn("2 kare", summary)

    def test_frame_size_comes_from_the_first_frame(self):
        factory = _Factory()
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, measure_seconds=0.1, writer_factory=factory)

        rec.add(_frame(width=320, height=240), timestamp=0.0)
        rec.add(_frame(width=320, height=240), timestamp=0.2)

        self.assertEqual(factory.made.size, (320, 240))

    def test_extension_picks_the_codec(self):
        avi = _Factory()
        OverlayRecorder("out.avi", warmup_seconds=0.0, measure_seconds=0.0, writer_factory=avi).add(
            _frame(), timestamp=0.0)
        # measure_seconds=0 ile ikinci kare açılışı tetikler
        rec = OverlayRecorder("out.avi", warmup_seconds=0.0, measure_seconds=0.0, writer_factory=avi)
        rec.add(_frame(), timestamp=0.0)
        rec.add(_frame(), timestamp=0.1)

        import cv2
        self.assertEqual(avi.made.fourcc, cv2.VideoWriter_fourcc(*"XVID"))

    def test_no_frames_at_all_is_reported_not_crashed(self):
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.0, writer_factory=_Factory())
        self.assertIn("boş", rec.close())

    def test_default_path_is_timestamped_mp4(self):
        path = default_path("/tmp")
        self.assertTrue(Path(path).as_posix().startswith("/tmp/astro_"))
        self.assertTrue(path.endswith(".mp4"))

    def test_warmup_frames_are_kept_but_do_not_set_the_rate(self):
        """The first half second is the slowest and must not decide the file's rate.

        Regression for a real one: a run that averaged 28 Hz was written at 15.8 fps
        because the measurement window started on the very first frame, while the camera
        was still warming up and the detector model was still loading. The recording
        then played back 1.8x slow — the opposite of what it exists to show, and just as
        misleading as playing back fast.
        """
        factory = _Factory()
        rec = OverlayRecorder("out.mp4", warmup_seconds=0.5, measure_seconds=0.5,
                              writer_factory=factory)

        # Half a second of slow warm-up (5 Hz), then a second at 25 Hz.
        t = 0.0
        for _ in range(3):
            rec.add(_frame(), timestamp=t)
            t += 0.2
        while t <= 1.6:
            rec.add(_frame(), timestamp=t)
            t += 0.04
        rec.close()

        self.assertAlmostEqual(rec.fps, 25.0, delta=3.0)
        # Nothing captured is thrown away, warm-up included.
        self.assertEqual(rec.frames_written, len(factory.made.frames))
        self.assertGreaterEqual(rec.frames_written, 3)

    def test_exit_during_warmup_still_produces_a_file(self):
        """A run shorter than the warm-up must not silently record nothing."""
        factory = _Factory()
        rec = OverlayRecorder("out.mp4", warmup_seconds=5.0, measure_seconds=1.0,
                              writer_factory=factory)

        rec.add(_frame(), timestamp=0.0)
        rec.add(_frame(), timestamp=0.1)
        rec.add(_frame(), timestamp=0.2)
        rec.close()

        self.assertEqual(rec.frames_written, 3)
        self.assertIsNotNone(rec.fps)


if __name__ == "__main__":
    unittest.main()
