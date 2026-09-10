#!/usr/bin/env python3
"""Terminale durum akışı — ekran yokken neler olduğunu okumak için.

Pencere açıkken bindirme şeridi zaten her şeyi söylüyor. Ekransız koşuda (`--no-window`,
yani robotun üzerinde koştuğu hal) geriye yalnızca çıkıştaki kare sayısı kalıyordu.

İki tür satır basılır:

  olay    — durum, sahip ya da hedef değiştiğinde. Teşhis burada: kafanın ne zaman
            neye dönmeye karar verdiğini gösterir.
  nabız   — sabit aralıkla. Hiçbir şey değişmezken bile açının nereye gittiğini ve
            programın yaşadığını gösterir.

Olay satırları her zaman basılır, nabız aralığı ne olursa olsun: kaçırılan bir durum
geçişi, kaçırılan bir nabızdan çok daha pahalıdır.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence


def _fmt_angle(value: Optional[float], width: int = 6) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:+{width}.1f}"


class StatusLog:
    """Takip döngüsünün durumunu satır satır yazar."""

    def __init__(
        self,
        interval_s: float = 1.0,
        printer: Callable[[str], None] = print,
    ):
        # interval_s <= 0: nabız kapalı, yalnızca olaylar basılır.
        self.interval_s = float(interval_s)
        self._print = printer
        self._last_beat: Optional[float] = None
        self._last_key: Optional[tuple] = None
        self._lines = 0

    @property
    def lines_written(self) -> int:
        return self._lines

    def update(
        self,
        elapsed_s: float,
        result,
        fps: float = 0.0,
        detections: int = 0,
        doa_deg: Optional[float] = None,
        head_feedback: bool = False,
        speech=None,
        fixed_head: bool = False,
    ) -> Optional[str]:
        """Gerekiyorsa bir satır basar ve bastığı satırı döndürür."""
        key = (
            getattr(result.gaze_state, "value", str(result.gaze_state)),
            getattr(result.owner, "value", str(result.owner)),
            result.target_id,
        )

        changed = key != self._last_key
        due = (
            self.interval_s > 0.0
            and (self._last_beat is None or (elapsed_s - self._last_beat) >= self.interval_s)
        )
        if not changed and not due:
            return None

        line = self._format(key, elapsed_s, result, fps, detections, doa_deg,
                            head_feedback, changed, speech, fixed_head)
        self._last_key = key
        if due or changed:
            self._last_beat = elapsed_s

        self._print(line)
        self._lines += 1
        return line

    @staticmethod
    def _speech_note(doa_deg, speech) -> str:
        """Kerterizin neden kabul ya da reddedildiğini tek kelimeyle söyler.

        Kafayı yalnızca konuşma çevirebiliyor. Bu, ekransız koşuda yeni bir sessiz
        arıza yolu açıyor: kerteriz var, kafa dönmüyor, sebep hiçbir yerde yazmıyor.
        README'nin teşhis tablosu hangi katmanın sustuğunu okumaya dayanıyor; bu
        sütun o tablonun ses tarafındaki karşılığı.
        """
        if speech is None:
            return "  [pencere yok]" if doa_deg is not None else ""
        if speech.is_speech:
            note = f"  [konusma {speech.confidence:.2f}]"
        else:
            note = f"  [elendi: {speech.reason}]"
        # Yön çıkmasa da konuşma penceresi ölçülmüş olabilir. Ret gerekçesini
        # sayılarla birlikte tutmak, gerçek cümleyi uzun 'aaa'dan ayırmayı sağlar.
        if all(hasattr(speech, field) for field in ("rms", "harmonicity", "modulation")):
            note += (f" rms={speech.rms:.5f} harm={speech.harmonicity:.3f}"
                     f" mod={speech.modulation:.3f}")
        return note

    def _format(self, key, elapsed_s, result, fps, detections, doa_deg,
                head_feedback, changed, speech=None, fixed_head=False) -> str:
        state, owner, target = key

        wanted = float(result.target_yaw_deg)
        actual = float(result.head_angle_deg)
        # Hata, komut ile ölçülen açı arasındaki fark: buyuk ve inatci ise sorun
        # aktuatorde ya da seri hatta, algilamada degil.
        error = wanted - actual

        marker = ">" if changed else " "
        pose_label = "sabit" if fixed_head else "gercek" if head_feedback else "tahmin"
        return (
            f"{marker}[{elapsed_s:6.1f}s] {state:<17} "
            f"{owner:<16} "
            f"hedef={target or '-':<12} "
            f"conf={result.confidence:4.2f}  "
            f"istenen{_fmt_angle(wanted)}  "
            f"{pose_label}{_fmt_angle(actual)}  "
            f"fark{_fmt_angle(error, 6)}  "
            f"ses{_fmt_angle(doa_deg, 6)}  "
            f"yuz={detections}  "
            f"{fps:4.1f}fps  "
            f"kafa:{'V' if head_feedback else 'X'}"
            f"{self._speech_note(doa_deg, speech)}"
        )

    def summary(self, elapsed_s: float, frames: int, bearings: Sequence[float] = ()) -> str:
        """Çıkışta tek satırlık kapanış."""
        hz = frames / elapsed_s if elapsed_s > 0 else 0.0
        return (f"— {frames} kare / {elapsed_s:.1f} sn = {hz:.1f} Hz, "
                f"{self._lines} durum satiri")
