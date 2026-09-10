#!/usr/bin/env python3
"""Kayıt: ekranda göremediğin şeyi sonradan izlemek için.

Kaydedilen kare, pencerede görünenin aynısıdır — kutular, hedef kimliği, istenen ve
gerçek açı dahil. Ham kamera görüntüsü kaydetmek burada işe yaramaz: sorun genelde
"kamera ne gördü" değil, "yığın onu ne sandı" olur, ve bunu yalnızca bindirme söyler.

Kare hızı dosya açılırken sabitlenir ve yanlış tahmin edilirse kayıt zamanlama
konusunda yalan söyler: 12 Hz'de koşan bir döngüyü 30 fps diye yazarsan video 2.5 kat
hızlı oynar ve teşhis etmeye çalıştığın takılma kaybolur. Bu yüzden yazıcı gecikmeli
açılır: ilk kareler tamponlanır, gerçekleşen hız ölçülür, dosya o hızla açılır.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

# Uzantıya göre codec. mp4v her yerde var; XVID .avi için daha güvenli.
_CODECS = {
    ".mp4": "mp4v",
    ".m4v": "mp4v",
    ".avi": "XVID",
    ".mkv": "mp4v",
}
_DEFAULT_CODEC = "mp4v"

# Ölçüm penceresi. Ama pencereyi ilk kareden başlatmak yanlış: ilk yarım saniye
# kameranın ısınması ve algılayıcı modelinin yüklenmesiyle geçiyor ve döngünün en
# yavaş olduğu andır. Ölçülen 28 Hz'lik bir koşu böyle 15.8 fps yazıldı ve video
# gerçeğin 1.8 katı yavaş oynadı — kaydın söylemesi gereken şeyin tam tersi.
# Isınma kareleri kayda giriyor, yalnızca hız hesabından dışlanıyor.
_WARMUP_SECONDS = 0.6
_MEASURE_SECONDS = 1.0

# Ölçüm saçmalarsa (kamera takıldı, tek kare geldi) kullanılacak sınırlar.
_MIN_FPS = 1.0
_MAX_FPS = 60.0
_FALLBACK_FPS = 15.0


def default_path(directory: str = ".") -> str:
    """Zaman damgalı bir dosya adı — üst üste koşarken kayıtlar birbirini ezmesin."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(Path(directory) / f"astro_{stamp}.mp4")


class OverlayRecorder:
    """Bindirilmiş kareleri videoya yazar.

    Kayıt teşhis amaçlıdır, zorunlu değil: yazıcı açılamazsa (codec yok, disk dolu,
    yol yazılamaz) program durmaz — kaydedici kendini kapatır ve sebebini `error`
    alanında söyler. Takip çalışmaya devam eder.
    """

    def __init__(
        self,
        path: str,
        measure_seconds: float = _MEASURE_SECONDS,
        warmup_seconds: float = _WARMUP_SECONDS,
        writer_factory=None,
    ):
        self.path = str(path)
        self.error: Optional[str] = None
        self.frames_written = 0
        self.fps: Optional[float] = None

        self._measure_seconds = max(0.0, float(measure_seconds))
        self._warmup_seconds = max(0.0, float(warmup_seconds))
        self._writer_factory = writer_factory or cv2.VideoWriter
        self._writer = None
        self._closed = False
        self._size: Optional[Tuple[int, int]] = None
        self._buffer: List = []
        self._first_ts: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._measure_from_ts: Optional[float] = None
        self._measure_from_index: int = 0

    @property
    def active(self) -> bool:
        """Hâlâ kare kabul ediyor mu (açılmayı bekliyor olabilir)."""
        return not self._closed and self.error is None

    def add(self, frame, timestamp: Optional[float] = None) -> None:
        """Bir bindirilmiş kare ver. Ölçüm bitene kadar tamponlanır."""
        if not self.active or frame is None:
            return

        now = time.monotonic() if timestamp is None else float(timestamp)

        if self._writer is not None:
            self._write(frame)
            self._last_ts = now
            return

        if self._first_ts is None:
            self._first_ts = now
            height, width = frame.shape[:2]
            self._size = (int(width), int(height))

        self._buffer.append(frame)
        self._last_ts = now

        # Warm-up frames are kept but do not count toward the rate.
        if self._measure_from_ts is None:
            if (now - self._first_ts) >= self._warmup_seconds:
                self._measure_from_ts = now
                self._measure_from_index = len(self._buffer) - 1
            return

        if (now - self._measure_from_ts) >= self._measure_seconds:
            self._open(self._measured_fps())

    def close(self) -> str:
        """Yazıcıyı kapatır ve insanca bir özet döndürür."""
        if self._closed:
            return self._summary()

        # Ölçüm süresi dolmadan çıkıldıysa elde olanla aç, yoksa tampon çöpe gider.
        if self._writer is None and self._buffer:
            self._open(self._measured_fps())

        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as exc:  # pragma: no cover - sürücüye bağlı
                self.error = self.error or f"yazıcı kapatılamadı: {exc}"

        self._closed = True
        return self._summary()

    # -- iç işler ---------------------------------------------------------

    def _measured_fps(self) -> float:
        """Isınma sonrası karelerden gerçekleşen hız."""
        start_ts = self._measure_from_ts if self._measure_from_ts is not None else self._first_ts
        start_i = self._measure_from_index

        if start_ts is None or self._last_ts is None:
            return _FALLBACK_FPS

        counted = len(self._buffer) - start_i
        span = self._last_ts - start_ts
        if counted < 2 or span <= 0:
            # Isınmayı geçemeden çıkıldı; elde ne varsa ondan kestir.
            if self._first_ts is None or len(self._buffer) < 2:
                return _FALLBACK_FPS
            span = self._last_ts - self._first_ts
            if span <= 0:
                return _FALLBACK_FPS
            return float(min(_MAX_FPS, max(_MIN_FPS, (len(self._buffer) - 1) / span)))

        # n kare arasında n-1 aralık var; n/span kullanmak hızı sistematik olarak
        # yüksek gösterir ve kısa tamponlarda fark belirgindir.
        fps = (counted - 1) / span
        return float(min(_MAX_FPS, max(_MIN_FPS, fps)))

    def _open(self, fps: float) -> None:
        if self._size is None:
            self.error = "kare boyutu bilinmiyor"
            return

        target = Path(self.path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.error = f"klasör oluşturulamadı: {exc}"
            self._buffer.clear()
            return

        codec = _CODECS.get(target.suffix.lower(), _DEFAULT_CODEC)
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = self._writer_factory(self.path, fourcc, fps, self._size)
        except Exception as exc:
            self.error = f"yazıcı açılamadı: {exc}"
            self._buffer.clear()
            return

        if not writer.isOpened():
            self.error = f"yazıcı açılamadı ({codec}, {self.path})"
            self._buffer.clear()
            return

        self._writer = writer
        self.fps = fps

        for frame in self._buffer:
            self._write(frame)
        self._buffer.clear()

    def _write(self, frame) -> None:
        try:
            self._writer.write(frame)
            self.frames_written += 1
        except Exception as exc:  # pragma: no cover - sürücüye bağlı
            self.error = f"kare yazılamadı: {exc}"

    def _summary(self) -> str:
        if self.error:
            return f"🎬 Kayıt yapılamadı: {self.error}"
        if self.frames_written == 0:
            return "🎬 Kayıt boş — hiç kare yazılmadı"
        seconds = self.frames_written / (self.fps or _FALLBACK_FPS)
        return (f"🎬 Kayıt: {self.path} "
                f"({self.frames_written} kare, {self.fps:.1f} fps, {seconds:.1f} sn)")
