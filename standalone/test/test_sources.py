#!/usr/bin/env python3
"""Tests for the camera and microphone sources.

Hardware is optional on purpose: this program has to be usable at a desk with no
Arduino and no ReSpeaker, otherwise it cannot be used to answer the question it
exists for. A missing device degrades the program, it does not stop it.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_path  # noqa: F401
from sources import (  # noqa: E402
    ARRAY_MIC_CHANNELS,
    RESPEAKER_MIC_CHANNELS,
    BLOCK_SAMPLES,
    DUPLICATE_BLOCKS_TO_LATCH,
    MIN_RMS,
    SAMPLE_RATE,
    UTTERANCE_BUFFER_S,
    AudioSource,
    CameraSource,
    to_detections,
)
from astro_audio.speech_detector import SpeechVerdict  # noqa: E402
from astro_audio.doa_estimator import (  # noqa: E402
    AcousticDOAEstimator,
    ReSpeakerGeometry,
)


class _FakeCapture:
    def __init__(self, frames=None, opened=True):
        self._frames = list(frames or [])
        self._opened = opened
        self.released = False
        self.props = {}

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        self.released = True


class TestDetectionConversion(unittest.TestCase):
    def test_detector_tuples_become_detections(self):
        found = to_detections([(10, 20, 30, 40, 0.91), (50, 60, 70, 80, 0.5)])

        self.assertEqual([(d.x, d.y, d.w, d.h) for d in found], [(10, 20, 30, 40), (50, 60, 70, 80)])
        self.assertAlmostEqual(found[0].confidence, 0.91)

    def test_nothing_found_is_an_empty_list_not_a_failure(self):
        self.assertEqual(to_detections([]), [])


class TestCameraSource(unittest.TestCase):
    def test_a_frame_is_handed_back_as_read(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        camera = CameraSource(capture=_FakeCapture([frame]))

        ok, out = camera.read()

        self.assertTrue(ok)
        self.assertEqual(out.shape, (480, 640, 3))

    def test_a_camera_that_will_not_open_is_reported_not_raised(self):
        camera = CameraSource(capture=_FakeCapture(opened=False))

        self.assertFalse(camera.available)
        self.assertEqual(camera.read(), (False, None))

    def test_detection_on_an_empty_frame_finds_nobody(self):
        camera = CameraSource(capture=_FakeCapture())

        self.assertEqual(camera.detect(np.zeros((480, 640, 3), np.uint8)), [])

    def test_closing_releases_the_device(self):
        capture = _FakeCapture()
        CameraSource(capture=capture).close()

        self.assertTrue(capture.released)


class TestAudioSource(unittest.TestCase):
    def test_without_a_microphone_there_is_simply_no_bearing(self):
        audio = AudioSource(stream_factory=lambda **_: (_ for _ in ()).throw(RuntimeError("no device")))

        audio.start()

        self.assertFalse(audio.available)
        self.assertIsNone(audio.latest_doa_deg(now=0.0))

    def test_a_bearing_is_offered_while_it_is_fresh(self):
        audio = AudioSource()
        audio._publish(doa_deg=42.0, timestamp=10.0)

        self.assertEqual(audio.latest_doa_deg(now=10.1), 42.0)

    def test_a_stale_bearing_is_withheld_rather_than_reused(self):
        """A DOA from a second ago says nothing about where the talker is now."""
        audio = AudioSource(max_age_s=0.5)
        audio._publish(doa_deg=42.0, timestamp=10.0)

        self.assertIsNone(audio.latest_doa_deg(now=11.0))

    def test_a_frame_of_silence_produces_no_bearing(self):
        audio = AudioSource()

        audio.process_block(np.zeros((512, 4), np.float32), timestamp=10.0)

        self.assertIsNone(audio.latest_doa_deg(now=10.0))

    def test_fewer_than_four_channels_cannot_be_localised(self):
        audio = AudioSource()

        audio.process_block(np.random.rand(512, 2).astype(np.float32) * 1000, timestamp=10.0)

        self.assertIsNone(audio.latest_doa_deg(now=10.0))


class _FakeStream:
    """Stands in for `sounddevice.InputStream`: records how it was opened."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class TestSampleRateIsNotAssumed(unittest.TestCase):
    """C1: the detector and estimator were built in `__init__` at a hardcoded
    16 kHz, before `start()` ever learns the device's real rate. In stereo mode
    that rate is the device's own — 44.1 kHz on this laptop, the mode this
    machine actually runs. Measured consequence (repo's own synthetic speech
    generator): analysed at the wrong rate, a 5 Hz syllable modulation aliases
    to 1.8 Hz (below the 4-8 Hz band actually used) and the 80-300 Hz pitch
    search actually scans 220-830 Hz -- speech is classified as not-speech, so
    the head can never turn toward a voice.
    """

    def test_stereo_mode_builds_detector_and_estimator_at_the_devices_rate(self):
        audio = AudioSource(stream_factory=lambda **kwargs: _FakeStream(**kwargs))
        # Gercek donanima dokunmadan stereo/44.1 kHz kosulunu kuruyor: tek bir
        # stereo (2 kanal) aygit, kendi varsayilan hizi 44100.
        audio._hardware_inputs = lambda: [
            (0, {"name": "Built-in Audio Analog Stereo",
                 "max_input_channels": 2, "default_samplerate": 44100.0})
        ]

        audio.start()

        self.assertTrue(audio.available, audio.error)
        self.assertEqual(audio.mode, "stereo")
        self.assertEqual(audio.sample_rate, 44100)
        self.assertEqual(
            audio._estimator.sample_rate, audio.sample_rate,
            "kestirici 44.1 kHz yakalamaya karsi 16 kHz'de kuruldu")
        self.assertEqual(
            audio._speech.sample_rate, audio.sample_rate,
            "konusma siniflandiricisi 44.1 kHz yakalamaya karsi 16 kHz'de kuruldu")


class _FakeEstimator:
    """Stands in for the GCC-PHAT estimator and records whether it was reached."""

    def __init__(self):
        self.calls = 0

    def estimate_from_multichannel_pcm(self, channels):
        self.calls += 1
        return 30.0, 0.9, True


class _FakeSpeechDetector:
    """Konusma siniflandiricisinin yerine gecer ve ne kadar ses gorduğunu kaydeder.

    Siniflandirmanin dogrulugu burada test edilmiyor -- o astro_audio'nun isi
    (test_speech_detector.py). Burada test edilen kablolama: pencere birikiyor mu,
    siniflandirici yeterince uzun bir pencereyle mi cagriliyor, verdi disari
    veriliyor mu.
    """

    def __init__(self, is_speech=True):
        self.is_speech = is_speech
        self.lengths = []

    def classify(self, mono):
        self.lengths.append(len(mono))
        return SpeechVerdict(is_speech=self.is_speech, confidence=0.8 if self.is_speech else 0.0,
                             harmonicity=0.6, modulation=0.8, rms=0.2)


def _block(amplitude, channels=4, samples=BLOCK_SAMPLES):
    """A block on the float32 scale the stream actually delivers: [-1.0, +1.0]."""
    rng = np.random.default_rng(1)
    return (rng.standard_normal((samples, channels)) * amplitude).astype(np.float32)


def _detached_source():
    """An AudioSource without opening a microphone."""
    import threading
    src = AudioSource.__new__(AudioSource)
    src._lock = threading.Lock()
    src._doa = None
    src._stamp = 0.0
    src.max_age_s = 0.5
    src._estimator = _FakeEstimator()
    src._speech = _FakeSpeechDetector()
    src._window = None
    src._verdict = None
    src._verdict_stamp = 0.0
    src.sample_rate = SAMPLE_RATE
    src._utterance = None
    src._written = 0
    src._array_dead = False
    src._duplicate_blocks = 0
    src._mic_channels = None
    src._requested_mic_channels = None
    src.available = True
    src.error = None
    return src


def _plane_wave_array(azimuth_deg, amplitude=0.15, samples=BLOCK_SAMPLES):
    """Four channels as a real ReSpeaker would deliver them, on the float32 scale.

    One noise burst reaching four capsules on a 43mm circle, each delayed by its own
    distance to the source. This is the input the estimator is built for, so it is the
    input that proves whether the plumbing in front of it is right.
    """
    radius = getattr(ReSpeakerGeometry, "HALF_SPACING_M", 0.032)
    speed = ReSpeakerGeometry.SPEED_OF_SOUND_MPS
    rng = np.random.default_rng(7)
    burst = (rng.standard_normal(samples) * amplitude).astype(np.float32)

    freqs = np.fft.rfftfreq(samples, 1.0 / SAMPLE_RATE)
    spectrum = np.fft.rfft(burst)
    heading = np.array([np.sin(np.radians(azimuth_deg)), np.cos(np.radians(azimuth_deg))])
    positions = [(-radius, 0.0), (0.0, -radius), (radius, 0.0), (0.0, radius)]

    channels = [
        np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * (-np.dot(p, heading) / speed)),
                     samples).astype(np.float32)
        for p in positions
    ]
    return np.stack(channels, axis=1)   # (samples, channels), as the stream delivers


def _six_channel_block(azimuth_deg):
    """Bir USB dizinin gerçekte verdiği blok: mikrofonlar ortada.

    Kanal 0 hüzme çıkışı (mikrofonların karışımı), 1-4 ham mikrofonlar, 5 geri
    besleme. Kanal 0'ın mikrofonlarla ilişkili ama gecikmesi anlamsız olması testin
    can alıcı noktası: bağımsız gürültü olsaydı kestirici zaten reddederdi, oysa
    gerçek arıza tam da bunun *inandırıcı* görünmesi.
    """
    mics = _plane_wave_array(azimuth_deg)
    beam = mics.sum(axis=1, keepdims=True) / 4.0
    rng = np.random.default_rng(7)
    playback = (rng.standard_normal((mics.shape[0], 1)) * 0.05).astype(np.float32)
    return np.concatenate([beam, mics, playback], axis=1)


def _real_estimator_source():
    """A detached source wired to the actual GCC-PHAT estimator."""
    src = _detached_source()
    src._estimator = AcousticDOAEstimator(sample_rate=SAMPLE_RATE)
    return src


class TestAudioEnergyGate(unittest.TestCase):
    """The gate that decides whether a block is worth localising.

    It exists because GCC-PHAT will happily return a bearing for room noise. But it
    has to be expressed on the same scale as the samples, and it was not: the stream
    is opened with dtype="float32", so samples arrive normalised to [-1.0, +1.0],
    while the threshold was 300.0 — a value from the int16 scale (+-32768). Nothing
    float32 can reach it, so every block was discarded, the estimator was never
    called, and no bearing ever existed for the head to turn toward.
    """

    def test_threshold_is_on_the_float32_scale(self):
        """A clipped full-scale block has an RMS of about 1.0. A threshold above that
        can never admit anything, which is exactly how this failed silently."""
        self.assertLess(MIN_RMS, 1.0,
                        "MIN_RMS is above full scale for float32 — nothing can pass it")

    def test_speech_level_block_reaches_the_estimator(self):
        src = _detached_source()
        src.process_block(_block(0.15), timestamp=1.0)

        self.assertEqual(src._estimator.calls, 1)
        self.assertEqual(src.latest_doa_deg(1.0), 30.0)

    def test_quiet_room_is_still_rejected(self):
        """The gate must keep doing its job: near-silence should not be localised."""
        src = _detached_source()
        src.process_block(_block(0.0005), timestamp=1.0)

        self.assertEqual(src._estimator.calls, 0)
        self.assertIsNone(src.latest_doa_deg(1.0))

    def test_fewer_than_four_channels_is_rejected(self):
        """No array geometry, no bearing — regardless of how loud it is."""
        src = _detached_source()
        src.process_block(_block(0.3, channels=2), timestamp=1.0)

        self.assertEqual(src._estimator.calls, 0)


class TestEstimatorScale(unittest.TestCase):
    """The estimator is shared with the ROS stack, which feeds it int16 PCM.

    Its energy gate (300.0) and its confidence term (rms / 1500.0) are both calibrated
    for that scale. This stream is float32, so a bearing was never returned even from a
    geometrically perfect array: the confidence term is structurally zero there — no
    float32 block can reach rms 1500 — which capped confidence below the 0.40 validity
    bar. Fixing MIN_RMS alone opened the outer gate onto an inner one that was shut for
    the same reason.
    """

    def test_a_real_array_yields_a_bearing(self):
        """The end-to-end claim: sound with genuine geometry produces a direction."""
        src = _real_estimator_source()
        src.process_block(_plane_wave_array(-40.0), timestamp=1.0)

        self.assertIsNotNone(
            src.latest_doa_deg(1.0),
            "a geometrically valid array produced no bearing — the estimator is being "
            "fed on a scale it was not calibrated for")

    def test_opposite_sides_are_told_apart(self):
        """Absolute convention needs the real device; distinguishing sides does not.

        The estimator's zero direction and rotation sense depend on how the hardware
        orders and times its channels, which cannot be settled without a ReSpeaker in
        hand. What can be settled here is that the bearing tracks the source at all.
        """
        left, right = _real_estimator_source(), _real_estimator_source()
        left.process_block(_plane_wave_array(-60.0), timestamp=1.0)
        right.process_block(_plane_wave_array(60.0), timestamp=1.0)

        a, b = left.latest_doa_deg(1.0), right.latest_doa_deg(1.0)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertGreater(abs(a - b), 30.0,
                           "left and right produced the same bearing")


class TestDuplicatedChannels(unittest.TestCase):
    """Opening four channels is not proof of four microphones.

    With no ReSpeaker attached the request still succeeds against PulseAudio's virtual
    `default` device, which satisfies it by duplicating the laptop's built-in stereo
    pair — channel 2 a copy of 0, channel 3 of 1, correlation 1.0000. Both GCC-PHAT
    delays are then exactly zero and the estimator returns a confident bearing straight
    down one axis, steady enough to pass for a real one. The head would hold a
    direction no sound came from.
    """

    def _duplicated_block(self, amplitude=0.15):
        rng = np.random.default_rng(3)
        left = (rng.standard_normal(BLOCK_SAMPLES) * amplitude).astype(np.float32)
        right = (rng.standard_normal(BLOCK_SAMPLES) * amplitude).astype(np.float32)
        return np.stack([left, right, left, right], axis=1)

    def test_duplicated_channels_produce_no_bearing(self):
        src = _real_estimator_source()
        for i in range(DUPLICATE_BLOCKS_TO_LATCH + 1):
            src.process_block(self._duplicated_block(), timestamp=1.0 + i * 0.01)

        self.assertIsNone(src.latest_doa_deg(1.1))

    def test_the_missing_array_is_reported_not_hidden(self):
        """A silent fallback to vision is fine; claiming the microphone works is not."""
        src = _real_estimator_source()
        for i in range(DUPLICATE_BLOCKS_TO_LATCH):
            src.process_block(self._duplicated_block(), timestamp=1.0 + i * 0.01)

        self.assertFalse(src.available)
        self.assertIsNotNone(src.error)

    def test_one_odd_block_does_not_condemn_the_array(self):
        """The latch needs repetition, so a single freak block cannot disable audio."""
        src = _real_estimator_source()
        src.process_block(self._duplicated_block(), timestamp=1.0)

        self.assertTrue(src.available)
        self.assertFalse(src._array_dead)

    def test_a_genuine_array_is_not_mistaken_for_duplicates(self):
        """The check must not fire on real microphones hearing the same sound."""
        src = _real_estimator_source()
        for i in range(DUPLICATE_BLOCKS_TO_LATCH + 2):
            src.process_block(_plane_wave_array(25.0), timestamp=1.0 + i * 0.01)

        self.assertTrue(src.available)
        self.assertIsNotNone(src.latest_doa_deg(1.0 + 0.01 * (DUPLICATE_BLOCKS_TO_LATCH + 1)))


class TestWhichChannelsAreMicrophones(unittest.TestCase):
    """İlan edilen kanal sayısı, kaç mikrofon olduğu anlamına gelmiyor.

    ReSpeaker USB Mic Array 6 kanal sunar: 0 işlenmiş hüzme, 1-4 ham mikrofonlar,
    5 geri besleme. İlk dördünü almak, kestiriciye "ön mikrofon" diye hüzme çıkışını
    verip dördüncü mikrofonu hiç okumamak demek — ve bu sessizce başarısız olur,
    çünkü hüzme kanalı mikrofonlarla ilişkilidir, gürültü değildir. Açı üretilmeye
    devam eder, sadece yanlıştır.
    """

    def _planned(self, available):
        source = AudioSource()
        opened = source._plan_channels({"max_input_channels": available})
        return source._mic_channels, opened

    def test_a_six_channel_array_reads_the_raw_microphones(self):
        wanted, opened = self._planned(6)

        self.assertEqual(wanted, RESPEAKER_MIC_CHANNELS)
        self.assertGreater(opened, max(wanted), "istenen kanal acilmiyor")

    def test_a_four_channel_array_reads_all_four(self):
        wanted, _ = self._planned(4)
        self.assertEqual(wanted, ARRAY_MIC_CHANNELS)

    def test_an_explicit_list_is_obeyed(self):
        """Dizi elde ölçülmüşse sıralamayı kullanıcı bilir, tahmin değil."""
        source = AudioSource(mic_channels=(4, 3, 2, 1))
        source._plan_channels({"max_input_channels": 6})

        self.assertEqual(source._mic_channels, (4, 3, 2, 1))

    TRUTHS = (-60.0, -30.0, 30.0, 60.0)

    def _arc(self, channels):
        """-60°'den +60°'ye tarayıp okunan kerteriz yayını döndürür.

        Kerteriz -180..+180 aralığına getirilerek döndürülür. `latest_doa_deg`
        0..360 yayınlıyor, ve o çerçevede iki uç arasındaki çıkarma sarmayı
        görmüyor: sol uçtaki -60, 300 olarak okunuyor ve 60-300 = -240 çıkıyor.
        Ölçülen yay yine 120°, sadece çıkarma yanlış çerçevede yapılıyordu.
        """
        bearings = []
        for truth in self.TRUTHS:
            source = _real_estimator_source()
            source._mic_channels = channels
            source.process_block(_six_channel_block(truth), timestamp=1.0)
            bearing = source.latest_doa_deg(1.0)
            bearings.append(None if bearing is None
                            else (bearing - 360.0 if bearing > 180.0 else bearing))
        return bearings

    def test_the_bearing_spans_the_arc_the_source_did(self):
        bearings = self._arc(RESPEAKER_MIC_CHANNELS)

        self.assertNotIn(None, bearings, "ham mikrofonlardan kerteriz cikmadi")
        self.assertAlmostEqual(bearings[-1] - bearings[0], 120.0, delta=15.0)

    def test_the_bearing_follows_the_source_instead_of_mirroring_it(self):
        """Yayı taramak yetmez: kerteriz kaynağın olduğu TARAFI göstermeli.

        Yay testi aynalanmış bir kestiriciyi yakalamıyor — soldan sağa tarayan bir
        kaynak, ters işaretli bir kestiricide de 120°'lik bir yay çizer, sadece
        ters yöne. Sahada bunun karşılığı kafanın konuşana değil tam tersine
        dönmesiydi. Burada her nokta ayrı ayrı karşılaştırılıyor.
        """
        bearings = self._arc(RESPEAKER_MIC_CHANNELS)

        for truth, measured in zip(self.TRUTHS, bearings):
            self.assertAlmostEqual(
                measured, truth, delta=15.0,
                msg=f"kaynak {truth:+.0f} derecede, kerteriz {measured:+.1f} dedi")

    def test_reading_the_beam_as_a_microphone_collapses_the_arc(self):
        """Asıl arıza buydu: açı üretiliyor ama kaynağı izlemiyor.

        120°'lik gerçek bir tarama, hüzme kanalı mikrofon sanıldığında 70°'nin altına
        sıkışıyor — ve sahada bu, açının rastgele sıçraması olarak görünüyor.
        """
        bearings = self._arc(ARRAY_MIC_CHANNELS)

        self.assertNotIn(None, bearings)
        self.assertLess(bearings[-1] - bearings[0], 90.0)


class TestSpeechWindow(unittest.TestCase):
    """Yon bloktan, konusma penceresinden okunur.

    Ikisi ayni surede olculemez. Yon 64 ms'lik bir bloktan cikar -- konusmaci
    hareket ederken daha uzun pencere kerterizi bulandirir. Hece modulasyonu ise
    tanimi geregi en az bir hece suresi ister; 4 Hz'lik bir dongu 250 ms. 64 ms'e
    bakip "modulasyon yok" demek olcum degil, pencerenin kisaligidir.

    Bu yuzden kaynak iki farkli zaman olceginde calisir: her blok bir kerteriz,
    biriken pencere bir konusma verdisi.
    """

    WINDOW_BLOCKS = 5   # 5 x 1024 ornek @16 kHz = 320 ms > MIN_BLOCK_S (300 ms)

    def _feed(self, src, blocks, t0=1.0):
        for i in range(blocks):
            src.process_block(_plane_wave_array(25.0), timestamp=t0 + i * 0.064)

    def test_tek_blok_konusma_hakkinda_karar_vermeye_yetmez(self):
        src = _detached_source()

        src.process_block(_plane_wave_array(25.0), timestamp=1.0)

        self.assertIsNone(src.latest_speech(1.0),
                          "64 ms'lik tek blokla konusma karari verildi")

    def test_pencere_dolunca_siniflandirici_calisir_ve_verdi_disari_verilir(self):
        src = _detached_source()

        self._feed(src, self.WINDOW_BLOCKS)

        verdict = src.latest_speech(1.0 + self.WINDOW_BLOCKS * 0.064)
        self.assertIsNotNone(verdict, "pencere doldu ama verdi yok")
        self.assertTrue(verdict.is_speech)

    def test_siniflandirici_hece_olcebilecek_uzunlukta_pencere_gorur(self):
        """Siniflandiriciya blok degil pencere gitmeli; yoksa 'modulasyon yok' der."""
        src = _detached_source()

        self._feed(src, self.WINDOW_BLOCKS)

        self.assertTrue(src._speech.lengths, "siniflandirici hic cagrilmadi")
        self.assertGreaterEqual(max(src._speech.lengths), int(0.30 * SAMPLE_RATE))

    def test_bayat_verdi_saklanmaz(self):
        """Bir saniye onceki konusma, simdi konusuldugunu soylemez."""
        src = _detached_source()

        self._feed(src, self.WINDOW_BLOCKS)

        self.assertIsNone(src.latest_speech(1.0 + self.WINDOW_BLOCKS * 0.064 + 5.0))


class TestUtteranceBuffer(unittest.TestCase):
    """Sozce tamponu konusma penceresinden ayri ve daha uzun.

    Konusma verdisi 0.6 s'lik pencereye bakiyor; sozce ise bittikten SONRA
    bastan sona geri okunmali. Ayni tamponu paylasirlarsa sozcenin basi silinir.
    """

    def test_yazilan_ses_geri_okunabiliyor(self):
        src = _detached_source()
        block = _plane_wave_array(20.0)

        src.process_block(block, timestamp=1.0)

        window = src.read_window(seconds=1.0)
        self.assertIsNotNone(window)
        self.assertEqual(window.ndim, 1, "sozce tamponu mono olmali")
        self.assertEqual(len(window), BLOCK_SAMPLES)

    def test_istenen_sureden_fazlasi_verilmez(self):
        src = _detached_source()
        for i in range(20):
            src.process_block(_plane_wave_array(20.0), timestamp=1.0 + i * 0.064)

        window = src.read_window(seconds=0.2)

        self.assertEqual(len(window), int(0.2 * SAMPLE_RATE))

    def test_tampon_konusma_penceresinden_uzun(self):
        """0.6 s'lik konusma penceresi bir sozceyi tutamaz."""
        src = _detached_source()
        for i in range(60):                       # 60 x 64 ms = 3.8 s
            src.process_block(_plane_wave_array(20.0), timestamp=1.0 + i * 0.064)

        window = src.read_window(seconds=3.0)

        self.assertGreater(len(window), int(2.0 * SAMPLE_RATE),
                           "tampon 0.6 s'lik konusma penceresi kadar kisa kalmis")

    def test_hic_ses_gelmemisse_none(self):
        self.assertIsNone(_detached_source().read_window(seconds=1.0))


class TestUtteranceCursor(unittest.TestCase):
    """Imlec her ornegi tam bir kez teslim eder.

    Ses dongusu ana dongudén 30 Hz'te (33 ms) pompalaniyor, bloklar ise 64 ms.
    "Son 64 ms"i okumak ardisik cagrilarda ortusur ve sozceye ayni sesi iki
    kez yazar; kelimeler kekeleyerek transkribe edilir.
    """

    def test_ayni_ses_iki_kez_teslim_edilmez(self):
        src = _detached_source()
        src.process_block(_plane_wave_array(20.0), timestamp=1.0)

        first, cursor = src.read_since(0)
        second, cursor2 = src.read_since(cursor)

        self.assertEqual(len(first), BLOCK_SAMPLES)
        self.assertEqual(len(second), 0, "ayni ses ikinci kez teslim edildi")
        self.assertEqual(cursor2, cursor)

    def test_yeni_ses_bir_sonraki_okumada_gelir(self):
        src = _detached_source()
        src.process_block(_plane_wave_array(20.0), timestamp=1.0)
        _, cursor = src.read_since(0)

        src.process_block(_plane_wave_array(20.0), timestamp=1.064)
        new, _ = src.read_since(cursor)

        self.assertEqual(len(new), BLOCK_SAMPLES)

    def test_hic_ses_yokken_bos_dizi_ve_ayni_imlec(self):
        empty, cursor = _detached_source().read_since(0)

        self.assertEqual(len(empty), 0)
        self.assertEqual(cursor, 0)

    def test_pompa_gecikirse_tamponun_tuttugu_kadari_verilir(self):
        """Imlec tamponun gerisinde kalirsa program durmaz, en eskisi dusurulur."""
        src = _detached_source()
        for i in range(200):                       # 12.8 sn > 10 sn tampon
            src.process_block(_plane_wave_array(20.0), timestamp=1.0 + i * 0.064)

        recovered, cursor = src.read_since(0)

        self.assertLessEqual(len(recovered), int(UTTERANCE_BUFFER_S * SAMPLE_RATE))
        self.assertEqual(cursor, 200 * BLOCK_SAMPLES)


if __name__ == "__main__":
    unittest.main()
