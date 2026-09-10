#!/usr/bin/env python3
"""Camera and microphone, read directly rather than through ROS topics.

Both detectors are the shared ones: YuNet via astro_vision.detection_quality, and
GCC-PHAT via astro_audio.doa_estimator. Nothing about perception is reimplemented
here — this file only opens the devices and hands their output on.

Neither device is required. Running with no ReSpeaker, or no camera, degrades the
program instead of stopping it, because a tool for answering "is it the algorithms
or the plumbing?" is worth little if it only runs on the finished robot.
"""

import os
import threading
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import depthai as dai
import numpy as np

import core_path  # noqa: F401
from astro_audio.doa_estimator import AcousticDOAEstimator  # noqa: E402
from astro_audio.speech_detector import MIN_BLOCK_S, SpeechDetector  # noqa: E402
from stereo_doa import DEFAULT_MIC_SPACING_M, StereoDOA  # noqa: E402
from astro_vision.detection_quality import create_face_detector  # noqa: E402

from tracker import Detection  # noqa: E402

SAMPLE_RATE = 16000
BLOCK_SAMPLES = 1024

# Konuşma penceresi. Yön ile konuşma aynı sürede ölçülemez: yön 64 ms'lik bir
# bloktan çıkar (uzun pencere, konuşmacı hareket ederken kerterizi bulandırır),
# hece modülasyonu ise en az bir hece süresi ister — 4 Hz'lik bir döngü 250 ms.
# 0.6 s iki-üç hece tutar; daha uzunu, susan birinin verdisini gereksiz uzatır.
SPEECH_WINDOW_S = 0.6

# Verdi de kerteriz gibi bayatlar. Bir saniye önce konuşulmuş olması şu an
# konuşulduğunu söylemez, ve kafayı çeviren yetki bayat bir kanıta dayanmamalı.
SPEECH_MAX_AGE_S = 0.5

# Sözce tamponu. Konuşma verdisi 0.6 s'lik pencereye bakar; sözcenin kendisi
# bittikten sonra baştan sona geri okunur, o yüzden ayrı ve uzun tutulur.
# 10 s, bir turda söylenebilecek en uzun cümleyi rahatça alır.
UTTERANCE_BUFFER_S = 10.0

# Below this the block is background noise and GCC-PHAT would localise the room.
#
# In float32 units. The stream is opened with dtype="float32", so sounddevice hands
# over samples normalised to [-1.0, +1.0] — this threshold has to live on that scale.
# It was 300.0, a number from the int16 scale (+-32768), which no float32 block can
# ever reach: a clipped full-scale signal has an RMS of 1.0. The gate therefore
# rejected every block including shouting, the estimator was never called, and no
# bearing was ever produced. The head could not turn toward a voice because it was
# never told there was one — while the status band still read `ses:V`, which only
# means the microphone opened.
MIN_RMS = 0.005

# The estimator is shared with the ROS audio stack, which feeds it int16 PCM: both its
# energy gate (300.0) and its confidence term (rms / 1500.0) are calibrated for that
# scale. This stream is float32, so samples arrive in [-1.0, +1.0] and both terms land
# near zero. The confidence term does so structurally — no float32 block can reach
# rms 1500 — which held confidence under the 0.40 validity bar even for a
# geometrically perfect array, so `valid` was never returned no matter the hardware.
# Scaling here fixes it without touching the estimator's other caller.
INT16_SCALE = 32768.0

# Correlation above this between two channels means one is a copy of the other rather
# than a second microphone. Real capsules never agree this closely; independent sensor
# and preamp noise keeps even co-located ones well below it.
DUPLICATE_CORRELATION = 0.9999
DUPLICATE_BLOCKS_TO_LATCH = 3

# ALSA her eklentisini bir aygıt gibi listeler. Bunlar mikrofon değil; 4 kanal istendiğinde
# kabul edip laptopun stereo çiftini kopyalayarak "başarılı" olan da bunlar.
VIRTUAL_DEVICE_NAMES = frozenset({
    "default", "sysdefault", "pulse", "pipewire", "samplerate", "speexrate",
    "upmix", "vdownmix", "dmix", "null", "jack", "oss", "spdif", "hdmi",
})
# Gerçek bir ses kartı bu kadar giriş kanalı sunmaz; eklentiler 32-128 ilan ediyor.
MAX_REAL_INPUT_CHANNELS = 16

# 44.1 kHz'de 4096 örnek ~93 ms: bir hecenin doğrudan yolunu tutacak kadar uzun,
# konuşmacı hareket ederken kerterizi bulandırmayacak kadar kısa.
STEREO_BLOCK_SAMPLES = 4096

# Bir USB mikrofon dizisinin ilan ettiği kanalların hepsi mikrofon değil. ReSpeaker
# USB Mic Array 6 kanal sunar: 0 işlenmiş/hüzmelenmiş ses, 1-4 ham mikrofonlar,
# 5 geri besleme. İlk dördünü almak, kestiriciye "ön mikrofon" diye hüzme çıkışını
# verip dördüncü mikrofonu hiç okumamak demek. Geometri baştan bozulur ve açı
# `atan2`'nin doygunluk değerlerine (0, ±90, 180) çakılır.
ARRAY_MIC_CHANNELS = (0, 1, 2, 3)
RESPEAKER_MIC_CHANNELS = (1, 2, 3, 4)


def _is_duplicate(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two channels carry the same signal."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return False
    return abs(float(np.dot(a, b)) / denom) >= DUPLICATE_CORRELATION


def to_detections(found) -> List[Detection]:
    """Turns the detector's (x, y, w, h, confidence) tuples into Detections."""
    return [
        Detection(x=int(x), y=int(y), w=int(w), h=int(h), confidence=float(conf))
        for (x, y, w, h, conf) in found
    ]


class CameraSource:
    """OAK-D Lite when one is attached, otherwise a plain webcam.

    The OAK-D is what the robot carries, so it is tried first. But this program has to
    stay runnable at a desk with no robot — a diagnostic tool that only starts on the
    finished machine cannot be used to diagnose the machine. So a missing OAK-D falls
    back to V4L2 rather than refusing to start, and `backend` says which one answered.
    """

    def __init__(
        self,
        device: int = 0,
        width: int = 640,
        height: int = 480,
        capture=None,
    ):
        self.pipeline = None
        self.queue = None
        self.device = None
        self.capture = capture
        self.available = False
        self.error = None
        self.backend = "none"

        if capture is not None:
            # Preserve the hardware-free/test injection path.
            self.available = bool(capture.isOpened())
            self.backend = "injected"
        else:
            try:
                self.pipeline = dai.Pipeline()

                cam = self.pipeline.create(dai.node.Camera).build()

                output = cam.requestOutput(
                    (width, height),
                    dai.ImgFrame.Type.BGR888p,
                    dai.ImgResizeMode.CROP,
                    30.0,
                )

                self.queue = output.createOutputQueue(
                    maxSize=2,
                    blocking=True,
                )

                self.pipeline.start()
                self.device = self.pipeline.getDefaultDevice()
                self.available = True
                self.backend = "OAK-D"

            except Exception as exc:
                # No OAK-D attached, or the pipeline would not build. Keep the reason —
                # it is the difference between "no camera plugged in" and "the OAK-D is
                # there but the pipeline is wrong", which are not the same problem.
                self.error = str(exc)
                self.available = False
                self.queue = None
                self.pipeline = None
                self.device = None

                fallback = cv2.VideoCapture(device)
                if fallback.isOpened():
                    fallback.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    fallback.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    fallback.set(cv2.CAP_PROP_FPS, 30.0)
                    self.capture = fallback
                    self.available = True
                    self.backend = "webcam"
                else:
                    fallback.release()

        model_dir = os.path.expanduser(
            os.getenv("FACE_MODEL_DIR", "~/.astro/models")
        )
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.detector = create_face_detector(model_dir, cascade)

    @property
    def detector_name(self) -> str:
        return type(self.detector).__name__

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.available:
            return False, None

        if self.capture is not None:
            return self.capture.read()

        try:
            frame = self.queue.get().getCvFrame()
            return True, frame
        except Exception:
            return False, None

    def detect(self, frame) -> List[Detection]:
        return to_detections(self.detector.detect(frame))

    def close(self) -> None:
        try:
            if self.capture is not None:
                self.capture.release()
        except Exception:
            pass

        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
class AudioSource:
    """A 4-channel microphone array reduced to one bearing at a time.

    The estimator needs the ReSpeaker's four channels; with fewer there is no
    geometry to solve and no bearing is offered. A bearing also expires: a DOA from
    a second ago says nothing about where the talker is now, and the fusion stage
    would rather have nothing than something stale.
    """

    min_rms: float = MIN_RMS
    min_harmonicity: float = 0.20
    min_modulation: float = 0.08
    doa_min_confidence: float = 0.25

    def __init__(
        self,
        device: Optional[int] = None,
        max_age_s: float = 1.0,
        stream_factory: Optional[Callable] = None,
        mic_spacing_m: float = DEFAULT_MIC_SPACING_M,
        mic_channels: Optional[Sequence[int]] = None,
        min_harmonicity: float = 0.20,
        min_modulation: float = 0.08,
        min_rms: float = MIN_RMS,
        doa_min_confidence: float = 0.25,
    ):
        self.device = device
        self.max_age_s = max_age_s
        self._stream_factory = stream_factory
        self.min_harmonicity = float(min_harmonicity)
        self.min_modulation = float(min_modulation)
        self.min_rms = float(min_rms)
        self.doa_min_confidence = float(doa_min_confidence)
        self._estimator = AcousticDOAEstimator(
            sample_rate=SAMPLE_RATE, min_confidence=self.doa_min_confidence
        )
        self._speech = SpeechDetector(
            sample_rate=SAMPLE_RATE,
            min_harmonicity=self.min_harmonicity,
            min_modulation=self.min_modulation,
            min_rms=self.min_rms,
        )
        self._window: Optional[np.ndarray] = None
        self._utterance: Optional[np.ndarray] = None
        self._written: int = 0        # tampona şimdiye kadar yazılan toplam örnek
        self._verdict = None
        self._verdict_stamp: float = 0.0
        self._lock = threading.Lock()
        self._doa: Optional[float] = None
        self._stamp: float = 0.0
        self._stream = None
        self.available = False
        self.error: Optional[str] = None
        self._array_dead = False
        self._duplicate_blocks = 0
        self.mode: Optional[str] = None
        self.device_name: Optional[str] = None
        self.sample_rate = SAMPLE_RATE
        self._stereo = None
        self._mic_spacing_m = mic_spacing_m
        self._requested_mic_channels = (
            tuple(int(c) for c in mic_channels) if mic_channels else None)
        self._mic_channels: Optional[tuple] = None

    def start(self) -> None:
        try:
            factory = self._stream_factory
            if factory is None:
                import sounddevice

                factory = sounddevice.InputStream

            device, channels, rate = self._choose_device()
            if device is None:
                self.error = "kullanilabilir mikrofon bulunamadi"
                self.available = False
                return

            self.mode = "array" if channels >= 4 else "stereo"
            self.sample_rate = rate
            # Hız `start()`'a kadar belli olmaz: dizi modunda 16 kHz sabit ama
            # stereo modda aygıtın kendi hızı kullanılıyor (laptopta 44.1 kHz).
            # `__init__`'te SAMPLE_RATE'e (16000) sabitlemek ikisini de yanlış
            # ölçekte kuruyordu — 44.1 kHz'lik 5 Hz'lik hece modülasyonu 1.8 Hz'e
            # (4-8 Hz bandının altına) düşüyor, 80-300 Hz'lik perde araması ise
            # aslında 220-830 Hz'i tarıyordu; konuşma "konuşma değil" sayılıyor ve
            # kafa hiçbir zaman bir sese dönemiyordu (ölçüldü, bkz. review C1).
            # `__init__`'teki varsayılan (16 kHz) donanımsız testler için duruyor;
            # burada gerçek hızla yeniden kuruluyor.
            self._estimator = AcousticDOAEstimator(
                sample_rate=rate, min_confidence=self.doa_min_confidence
            )
            self._speech = SpeechDetector(
                sample_rate=rate,
                min_harmonicity=self.min_harmonicity,
                min_modulation=self.min_modulation,
                min_rms=self.min_rms,
            )
            if self.mode == "stereo":
                self._stereo = StereoDOA(sample_rate=rate,
                                         mic_spacing_m=self._mic_spacing_m)

            self._stream = factory(
                device=device, channels=channels, samplerate=rate,
                blocksize=BLOCK_SAMPLES if self.mode == "array" else STEREO_BLOCK_SAMPLES,
                dtype="float32", callback=self._on_block,
            )
            self._stream.start()
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def _choose_device(self):
        """Picks a real sound card, and the best mode it can support.

        Asking the `default` device for four channels is what hid the missing array:
        ALSA's plugin layer accepts the request and fills it by duplicating the
        laptop's stereo pair. So the device is chosen before the channel count is,
        and only from hardware.

        A genuine four-channel card gives the full circle. Two channels give one
        axis — front/back ambiguous, but enough to decide which way to turn, and the
        built-in pair is genuinely separated: measured -3.99 samples from the left
        speaker against +4.19 from the right, at 44.1 kHz on the raw device.

        Every channel the card offers is opened, because which of them are microphones
        is a separate question from how many there are.
        """
        if self.device is not None:
            info = self._device_info(self.device)
            self.device_name = info["name"]
            return (self.device, self._plan_channels(info),
                    int(info["default_samplerate"]))

        best_stereo = None
        for index, info in self._hardware_inputs():
            if info["max_input_channels"] >= 4:
                self.device_name = info["name"]
                return index, self._plan_channels(info), SAMPLE_RATE
            if best_stereo is None and info["max_input_channels"] >= 2:
                best_stereo = (index, info)

        if best_stereo is None:
            return None, 0, SAMPLE_RATE

        index, info = best_stereo
        self.device_name = info["name"]
        return index, self._plan_channels(info), int(info["default_samplerate"])

    def _plan_channels(self, info) -> int:
        """Kaç kanal açılacağını ve bunların hangilerinin mikrofon olduğunu belirler."""
        available = int(info["max_input_channels"])
        if available < 4:
            self._mic_channels = None
            return 2

        # Açıkça verilmişse tartışma yok: kullanıcı diziyi elinde ölçmüştür.
        if self._requested_mic_channels is not None:
            self._mic_channels = tuple(self._requested_mic_channels)
            return max(self._mic_channels) + 1

        if available >= 6:
            # 6 kanallı USB dizilerin yerleşimi: baştaki işlenmiş, sondaki geri
            # besleme. Ham mikrofonlar aradadır.
            self._mic_channels = RESPEAKER_MIC_CHANNELS
            return max(RESPEAKER_MIC_CHANNELS) + 1

        self._mic_channels = ARRAY_MIC_CHANNELS
        return 4

    @staticmethod
    def _device_info(index):
        import sounddevice

        return sounddevice.query_devices(index)

    @staticmethod
    def _hardware_inputs():
        """Real capture devices, with ALSA's plugin aliases filtered out."""
        import sounddevice

        found = []
        for index, info in enumerate(sounddevice.query_devices()):
            inputs = info["max_input_channels"]
            if inputs < 1 or inputs > MAX_REAL_INPUT_CHANNELS:
                continue
            if info["name"].split(":")[0].strip().lower() in VIRTUAL_DEVICE_NAMES:
                continue
            found.append((index, info))
        return found

    def _on_block(self, indata, _frames, time_info, _status):
        import time as _time

        block = np.array(indata, copy=True)
        now = _time.monotonic()
        if self.mode == "stereo":
            self.process_stereo_block(block, timestamp=now)
        else:
            self.process_block(block, timestamp=now)

    def process_stereo_block(self, block: np.ndarray, timestamp: float) -> None:
        """Localises one block from a two-microphone pair."""
        if self._stereo is None or block.ndim != 2:
            return
        channels = block.T if block.shape[0] > block.shape[1] else block
        if channels.shape[0] < 2:
            return

        self._observe_speech(channels[0], timestamp)

        if float(np.sqrt(np.mean(np.square(channels)))) < self.min_rms:
            return

        azimuth, _sharpness = self._stereo.estimate(channels[0], channels[1])
        if azimuth is not None:
            self._publish(azimuth if azimuth >= 0.0 else azimuth + 360.0, timestamp)

    def process_block(self, block: np.ndarray, timestamp: float) -> None:
        """Localises one block, if it has the channels and the energy to justify it."""
        if block.ndim != 2 or min(block.shape) < 4:
            return
        channels = block.T if block.shape[0] > block.shape[1] else block

        wanted = self._mic_channels or ARRAY_MIC_CHANNELS
        if channels.shape[0] <= max(wanted):
            return
        # Yalnızca mikrofonlar. İşlenmiş ve geri besleme kanalları hem geometriyi
        # bozar hem de enerji ölçüsünü yanıltır.
        mics = channels[list(wanted)]

        # Konuşma sınıflandırmasında dairesel mikrofonların ortalaması (mics.mean)
        # faz gecikmesi sebebiyle 1-3 kHz bandında yıkıcı faz girişimine (comb filtering)
        # yol açar. Eğer 6 kanallı ReSpeaker DSP kanalı (channels[0]) varsa donanımsal
        # filtrelenmiş sesi, yoksa ilk ham mikrofonu (mics[0]) kullan.
        mono_speech = channels[0] if channels.shape[0] >= 6 else mics[0]
        self._observe_speech(mono_speech, timestamp)

        if float(np.sqrt(np.mean(np.square(mics)))) < self.min_rms:
            return

        if not self._array_is_real(mics):
            return

        azimuth, _confidence, valid = self._estimator.estimate_from_multichannel_pcm(
            mics * INT16_SCALE
        )
        if valid and azimuth is not None:
            self._publish(azimuth if azimuth >= 0.0 else azimuth + 360.0, timestamp)

    def _array_is_real(self, channels: np.ndarray) -> bool:
        """Latches false when the four channels are two channels duplicated.

        Opening a 4-channel stream is not proof of a 4-microphone array. With no
        ReSpeaker attached the request still succeeds against PulseAudio's virtual
        `default` device, which satisfies it by duplicating the laptop's built-in
        stereo pair: channel 2 becomes a copy of channel 0 and channel 3 of channel 1.
        GCC-PHAT then measures a delay of exactly zero on both axes and returns a
        confident bearing straight down one of them — a direction no sound came from,
        held steady enough to look real. Refusing here makes the missing hardware
        visible instead of aiming the head at it.
        """
        if self._array_dead:
            return False

        if (_is_duplicate(channels[0], channels[2])
                or _is_duplicate(channels[1], channels[3])):
            self._duplicate_blocks += 1
            if self._duplicate_blocks >= DUPLICATE_BLOCKS_TO_LATCH:
                self._array_dead = True
                self.available = False
                self.error = ("4 kanal açıldı ama iki kanalın kopyası — mikrofon "
                              "dizisi bağlı değil, yön bulma kapatıldı")
            return False

        self._duplicate_blocks = 0
        return True

    def _observe_speech(self, mono: np.ndarray, timestamp: float) -> None:
        """Bloğu kayan pencereye ekler ve pencere yeterince uzunsa sınıflandırır.

        Pencere **her** blokla beslenir, enerji kapısına bakılmadan. Sebebi ölçünün
        kendisi: konuşmayı ayıran şey hecelerin arasındaki susmalar, yani zarfın
        4-8 Hz'te açılıp kapanması. Yalnızca yüksek blokları biriktirmek tam da o
        susmaları siler ve geriye kesintisiz bir uğultu bırakır — sınıflandırıcıya
        gerçek konuşmayı "modülasyonsuz" diye gösterirdi.
        """
        self._append_utterance(mono)

        capacity = int(SPEECH_WINDOW_S * self.sample_rate)
        block = np.asarray(mono, dtype=np.float32).ravel()
        self._window = (block if self._window is None
                        else np.concatenate((self._window, block)))[-capacity:]

        if len(self._window) < int(MIN_BLOCK_S * self.sample_rate):
            return

        verdict = self._speech.classify(self._window)
        with self._lock:
            self._verdict, self._verdict_stamp = verdict, float(timestamp)

    def _append_utterance(self, mono: np.ndarray) -> None:
        """Sözce tamponunu besler. Konuşma penceresinden ayrı tutulur."""
        capacity = int(UTTERANCE_BUFFER_S * self.sample_rate)
        block = np.asarray(mono, dtype=np.float32).ravel()
        with self._lock:
            self._utterance = (block if self._utterance is None
                               else np.concatenate((self._utterance, block)))[-capacity:]
            self._written += len(block)

    def read_since(self, cursor: int) -> Tuple[np.ndarray, int]:
        """`cursor`'dan sonra yazılan yeni örnekler ve yeni imleç.

        Ses döngüsü ana döngüden 30 Hz'te (33 ms) pompalanıyor, bloklar ise 64 ms.
        "Son N saniye" okumak ardışık çağrılarda örtüşür ve sözceye aynı sesi iki
        kez yazar. İmleç her örneği tam bir kez teslim eder.

        İmleç tamponun gerisinde kalırsa (pompa geciktiyse) elde ne varsa verilir:
        eksik ses, duran bir program'dan iyidir.
        """
        with self._lock:
            if self._utterance is None:
                return np.zeros(0, dtype=np.float32), cursor
            held = len(self._utterance)
            oldest = self._written - held
            start = max(0, int(cursor) - oldest)
            if start >= held:
                return np.zeros(0, dtype=np.float32), self._written
            return np.array(self._utterance[start:], copy=True), self._written

    def read_window(self, seconds: float) -> Optional[np.ndarray]:
        """Son `seconds` saniyelik mono ses. Hız `self.sample_rate`.

        16 kHz garanti edilmez: dizi modunda 16 kHz açılıyor ama stereo modda
        aygıtın kendi hızı kullanılıyor (laptopta 44.1 kHz). Dönüştürme burada
        değil, STT'ye verilmeden hemen önce tek noktada yapılır.
        """
        with self._lock:
            if self._utterance is None or len(self._utterance) == 0:
                return None
            wanted = max(1, int(seconds * self.sample_rate))
            return np.array(self._utterance[-wanted:], copy=True)

    def latest_speech(self, now: float):
        """En son konuşma verdisi, hâlâ tazeyse. Yoksa None."""
        with self._lock:
            if self._verdict is None or (now - self._verdict_stamp) > SPEECH_MAX_AGE_S:
                return None
            return self._verdict

    def _publish(self, doa_deg: float, timestamp: float) -> None:
        with self._lock:
            self._doa, self._stamp = float(doa_deg), float(timestamp)

    def latest_doa_deg(self, now: float) -> Optional[float]:
        with self._lock:
            if self._doa is None or (now - self._stamp) > self.max_age_s:
                return None
            return self._doa

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
