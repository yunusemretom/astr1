#!/usr/bin/env python3
"""Mikrofon dizisini ölçüp doğru kanal sırasını bulur.

Kaç kanal olduğu ile hangilerinin mikrofon olduğu ayrı sorular, ve mikrofonların
hangi sırayla geldiği üçüncü bir soru. Kestirici kanalları (ön, sağ, arka, sol)
diye okur; dizi başka sırayla veriyorsa açılar döner ya da aynalanır ve bu, sahada
"açı saçmalıyor" diye görünür — üretilen değer inandırıcıdır, sadece yanlıştır.

Burası tahmin etmiyor: dört bilinen yönden ses kaydedip 24 sıralamanın hepsini
deniyor ve ölçümle en tutarlı olanı söylüyor.

    python standalone/audio_check.py

Sonuçta verilen `--mic-channels` değerini `track.py`'ye geçirin.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core_path  # noqa: F401,E402
from astro_audio.doa_estimator import AcousticDOAEstimator  # noqa: E402
from sources import INT16_SCALE, SAMPLE_RATE  # noqa: E402

# Kestiricinin sözleşmesi: 0° ön, 90° sağ, 180° arka, 270° sol.
#
# Çaprazlar süs değil, ölçümün ayırt edebilmesi için şart. Tam ana yönlerde
# `atan2`'nin bir ekseni sıfırdır ve açı o eksendeki kazanca duyarsız kalır: hüzme
# kanalı bir mikrofonun yerine konsa bile sonuç değişmez, ve arama beş sıralamayı
# 0.0° hatayla berabere bulur. Çapraz yönde iki eksen de çalışır, sahte sıralamalar
# ayrışır.
DIRECTIONS = (
    ("ÖN", 0.0),
    ("ÖN-SAĞ ÇAPRAZ", 45.0),
    ("SAĞ", 90.0),
    ("ARKA", 180.0),
    ("ARKA-SOL ÇAPRAZ", 225.0),
    ("SOL", 270.0),
)
BLOCK = 4096
RECORD_SECONDS = 3.0
BLOCKS_PER_DIRECTION = 6


def _angular_error(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _loudest_blocks(recording: np.ndarray, count: int):
    """En yüksek enerjili blokları seçer — aradaki sessizlik yönü bulandırır."""
    blocks = [recording[i:i + BLOCK] for i in range(0, len(recording) - BLOCK, BLOCK)]
    blocks.sort(key=lambda b: float(np.mean(np.square(b))), reverse=True)
    return blocks[:count]


def _capture(device, channels, rate):
    import sounddevice as sd

    captured = {}
    for label, truth in DIRECTIONS:
        for remaining in (3, 2, 1):
            print(f"\r  {label} yönünden konuşmaya hazırlanın... {remaining}", end="", flush=True)
            time.sleep(1.0)
        print(f"\r  {label} yönünden konuşun ({RECORD_SECONDS:.0f} sn)          ", flush=True)

        recording = sd.rec(int(RECORD_SECONDS * rate), samplerate=rate,
                           channels=channels, dtype="float32", device=device)
        sd.wait()
        captured[truth] = _loudest_blocks(recording, BLOCKS_PER_DIRECTION)
    return captured


def _score(estimator, captured, order):
    """Bir kanal sıralamasının ölçümle ne kadar uyuştuğu: küçük olan iyi."""
    errors = []
    for truth, blocks in captured.items():
        for block in blocks:
            mics = block[:, list(order)].T * INT16_SCALE
            azimuth, _confidence, valid = estimator.estimate_from_multichannel_pcm(mics)
            if valid and azimuth is not None:
                errors.append(_angular_error(azimuth % 360.0, truth))
    if not errors:
        return None, 0
    return float(np.mean(errors)), len(errors)


def main() -> int:
    import sounddevice as sd

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=None, help="Aygıt numarası")
    parser.add_argument("--channels", type=int, default=None,
                        help="Açılacak kanal sayısı (varsayılan: aygıtın tamamı)")
    opts = parser.parse_args()

    device = opts.device
    if device is None:
        for index, info in enumerate(sd.query_devices()):
            name = info["name"].lower()
            if info["max_input_channels"] >= 4 and "respeaker" in name:
                device = index
                break
    if device is None:
        print("4+ kanallı bir mikrofon dizisi bulunamadı. --device ile verin.")
        print("\nGiriş aygıtları:")
        for index, info in enumerate(sd.query_devices()):
            if info["max_input_channels"] > 0:
                print(f"  [{index}] {info['name']}  kanal={info['max_input_channels']}")
        return 1

    info = sd.query_devices(device)
    channels = opts.channels or int(info["max_input_channels"])
    rate = SAMPLE_RATE
    print(f"Aygıt [{device}] {info['name']}")
    print(f"{channels} kanal @ {rate} Hz açılıyor\n")

    captured = _capture(device, channels, rate)

    print("\nKanal enerjileri (ölü ya da kopya kanal buradan görünür):")
    stacked = np.concatenate([b for blocks in captured.values() for b in blocks])
    for channel in range(channels):
        print(f"  kanal {channel}: RMS {float(np.sqrt(np.mean(np.square(stacked[:, channel])))):.5f}")

    estimator = AcousticDOAEstimator(sample_rate=rate)
    candidates = [c for c in range(channels)]
    results = []
    for order in itertools.permutations(candidates, 4):
        mean_error, samples = _score(estimator, captured, order)
        if mean_error is not None and samples >= 4:
            results.append((mean_error, samples, order))

    if not results:
        print("\nHiçbir sıralama geçerli kerteriz üretmedi — ses çok mu kısıktı?")
        return 1

    results.sort()
    print("\nEn iyi beş sıralama (ortalama hata):")
    for mean_error, samples, order in results[:5]:
        print(f"  {','.join(str(c) for c in order):<12} {mean_error:6.1f}°  ({samples} ölçüm)")

    best_error, _, best = results[0]
    print(f"\n>>> Kullanın:  --mic-channels {','.join(str(c) for c in best)}")
    if len(results) > 1 and (results[1][0] - best_error) < 5.0:
        print(f"    Not: ikinci sıralama yalnızca {results[1][0] - best_error:.1f}° geride.")
        print("    Ayrım zayıf — daha yüksek sesle ve yankısı az bir odada tekrarlayın.")
    if best_error > 45.0:
        print(f"    Ama en iyi sıralama bile {best_error:.0f}° hata veriyor. Yönleri")
        print("    karıştırmış olabilirsiniz, ya da oda çok yankılı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
