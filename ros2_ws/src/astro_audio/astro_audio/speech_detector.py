#!/usr/bin/env python3
"""Bir ses bloğunun insan konuşması olup olmadığına karar verir.

Kafayı çevirme yetkisi buraya bağlanır. Sebep ölçülmüş bir arıza: sahadaki bir
koşuda kafa 90 saniye boyunca ±75 arasında salındı, çünkü kerteriz üreten her
yüksek ses hedefi ele geçirebiliyordu. Enerji bunu elemez — trafik de yüksektir.
Yön kararlılığı da elemez — geçen araba da, sabit bir uğultu da kararlıdır;
kararlılık şartı onları eler değil, **ödüllendirir**.

Ayrım sinyalin kendisinde ve iki ölçü gerektiriyor, biri yetmiyor:

    harmoniklik   sesli konuşmanın perdesi vardır (F0 ~80-300 Hz). Araba
                  gürültüsü harmonik değildir — ama tonal bir uğultu öyledir.
    modülasyon    konuşma hece hızında (4-8 Hz) açılıp kapanır; nefes alır.
                  Uğultu almaz, araba gürültüsü de almaz (geçerken şişer, ama
                  ~0.5 Hz'te — hece bandının çok altında).

Ölçülen ayrım (test/test_speech_detector.py sentetik sahneleri, sahnenin kendisi
orada bağımsız olarak doğrulanıyor):

    konuşma   harmoniklik 0.59   hece modülasyonu 0.83
    araba     harmoniklik 0.32   hece modülasyonu 0.03
    uğultu    harmoniklik 0.99   hece modülasyonu 0.00

Uğultu satırı bu modülün neden iki ölçüye birden baktığını tek başına açıklıyor:
harmonikliğe bakıp modülasyona bakmayan bir sınıflandırıcı ona kafayı çevirir.

Tek kanallıdır — mikrofon dizisi gerektirmez. Yön ayrı bir soru (doa_estimator);
burada yalnızca "bu ses bir insan mı" sorulur.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# İnsan perde aralığı. Alt sınır kalın erkek sesini, üst sınır çocuk/tiz kadın
# sesini kapsıyor; dışına çıkmak uğultuları perde arayışına dahil etmek demek.
MIN_PITCH_HZ = 80.0
MAX_PITCH_HZ = 300.0

# Hece hızı bandı. Konuşma zarfının enerjisi buraya yığılır.
MIN_SYLLABLE_HZ = 4.0
MAX_SYLLABLE_HZ = 8.0

# Zarf, bu örnekleme hızına indirgenerek incelenir: hece bandı için fazlasıyla
# yeterli (Nyquist 100 Hz) ve blok başına maliyeti önemsiz kılıyor.
ENVELOPE_RATE_HZ = 200.0

# Eşikler gerçek ortam akustiği ve sentetik sahneler arasındaki ayrımı koruyacak
# şekilde ayarlandı: konuşma (~0.25+ / 0.10+), trafik (~0.32 / 0.03 -> modülasyondan elenir),
# sabit uğultu (~0.99 / 0.00 -> modülasyondan elenir).
DEFAULT_MIN_HARMONICITY = 0.25
DEFAULT_MIN_MODULATION = 0.10

# Altında blok zaten sessiz sayılır ve yön iddia edilmez. float32 ölçeğinde.
DEFAULT_MIN_RMS = 0.005

# Perde ve hece ölçülerinin anlamlı olması için gereken en kısa blok. 4 Hz'lik bir
# hece döngüsü 250 ms sürer; bunun altında "modülasyon yok" demek ölçüm değil,
# pencerenin kısalığıdır.
MIN_BLOCK_S = 0.30


@dataclass
class SpeechVerdict:
    """Bir blok icin karar ve onu ureten iki olcu.

    Ölçüler kararla birlikte taşınıyor çünkü bu yığında bir davranışı açıklamak
    için hangi katmanın sustuğunu görmek gerekiyor: "konuşma sayılmadı" tek
    başına teşhis değil, "harmonikti ama modülasyonsuzdu" teşhistir.
    """
    is_speech: bool
    confidence: float
    harmonicity: float
    modulation: float
    rms: float
    reason: str = ""


class SpeechDetector:
    """Tek kanallı bloklardan konuşma/konuşma-değil kararı."""

    def __init__(
        self,
        sample_rate: int = 16000,
        min_harmonicity: float = DEFAULT_MIN_HARMONICITY,
        min_modulation: float = DEFAULT_MIN_MODULATION,
        min_rms: float = DEFAULT_MIN_RMS,
    ):
        self.sample_rate = int(sample_rate)
        self.min_harmonicity = float(min_harmonicity)
        self.min_modulation = float(min_modulation)
        self.min_rms = float(min_rms)

    def classify(self, mono_pcm: np.ndarray) -> SpeechVerdict:
        """Bir ses bloğunu sınıflandırır. Giriş float32 ya da int16 olabilir."""
        samples = np.asarray(mono_pcm, dtype=np.float64).ravel()
        if samples.size == 0:
            return SpeechVerdict(False, 0.0, 0.0, 0.0, 0.0, "bos blok")

        # int16 gelirse float32 ölçeğine indir; eşikler o ölçekte tanımlı.
        if np.max(np.abs(samples)) > 2.0:
            samples = samples / 32768.0

        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < self.min_rms:
            return SpeechVerdict(False, 0.0, 0.0, 0.0, rms, "sessiz")

        if samples.size < int(MIN_BLOCK_S * self.sample_rate):
            return SpeechVerdict(False, 0.0, 0.0, 0.0, rms, "blok cok kisa")

        harmonicity = self._harmonicity(samples)
        modulation = self._syllable_modulation(samples)

        harmonic_enough = harmonicity >= self.min_harmonicity
        modulated_enough = modulation >= self.min_modulation
        is_speech = harmonic_enough and modulated_enough

        if is_speech:
            reason = ""
        elif harmonic_enough:
            # Uğultu tam buraya düşer: perdesi var, nefesi yok.
            reason = "harmonik ama hece modulasyonu yok"
        elif modulated_enough:
            reason = "modulasyonlu ama perdesiz"
        else:
            reason = "ne harmonik ne modulasyonlu"

        return SpeechVerdict(
            is_speech=is_speech,
            confidence=self._confidence(harmonicity, modulation) if is_speech else 0.0,
            harmonicity=round(harmonicity, 3),
            modulation=round(modulation, 3),
            rms=round(rms, 5),
            reason=reason,
        )

    def _harmonicity(self, samples: np.ndarray) -> float:
        """İnsan perde aralığındaki normalize otokorelasyon tepesi [0..1].

        Perdesi olan bir ses kendi periyodunda kendine benzer; gürültü benzemez.
        Arama aralığı insan perdesiyle sınırlı, yoksa çok alçak bir uğultunun
        periyodu da tepe verir.
        """
        centred = samples - samples.mean()
        energy = float(np.dot(centred, centred))
        if energy <= 0.0:
            return 0.0

        # FFT ile otokorelasyon: 4096 orneklik blokta dogrudan korelasyondan
        # belirgin olcude ucuz, sonuc ayni.
        n = 1 << int(np.ceil(np.log2(len(centred) * 2)))
        spectrum = np.fft.rfft(centred, n)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum), n)[: len(centred)]
        correlation = correlation / correlation[0]

        lo = max(1, int(self.sample_rate / MAX_PITCH_HZ))
        hi = min(len(correlation) - 1, int(self.sample_rate / MIN_PITCH_HZ))
        if hi <= lo:
            return 0.0
        return float(np.clip(np.max(correlation[lo:hi]), 0.0, 1.0))

    def _syllable_modulation(self, samples: np.ndarray) -> float:
        """Zarf enerjisinin hece bandına düşen oranı [0..1].

        Zarf, mutlak değerin sabit pencerelerle ortalanmasıyla çıkarılır: hece
        hızı için yeterli, ve Hilbert dönüşümünden ucuz.
        """
        step = max(1, int(self.sample_rate / ENVELOPE_RATE_HZ))
        usable = (len(samples) // step) * step
        if usable < step * 16:
            return 0.0

        envelope = np.abs(samples[:usable]).reshape(-1, step).mean(axis=1)
        envelope = envelope - envelope.mean()
        if not np.any(envelope):
            return 0.0

        spectrum = np.abs(np.fft.rfft(envelope)) ** 2
        freqs = np.fft.rfftfreq(len(envelope), step / self.sample_rate)
        total = float(spectrum.sum())
        if total <= 0.0:
            return 0.0

        band = float(spectrum[(freqs >= MIN_SYLLABLE_HZ) & (freqs <= MAX_SYLLABLE_HZ)].sum())
        return float(np.clip(band / total, 0.0, 1.0))

    def _confidence(self, harmonicity: float, modulation: float) -> float:
        """İki ölçünün eşiğin ne kadar üstünde olduğundan türeyen güven [0..1].

        Kestiricinin kendi 'güven'i kullanılmıyor: ölçüldüğünde dört farklı
        akustik koşulda 0.40-0.46 arasında kaldı, yani iyi yönü kötüden ayırmıyor,
        yalnızca sesin yüksekliğini söylüyor.
        """
        h_margin = (harmonicity - self.min_harmonicity) / max(1e-6, 1.0 - self.min_harmonicity)
        m_margin = (modulation - self.min_modulation) / max(1e-6, 1.0 - self.min_modulation)
        score = 0.5 * np.clip(h_margin, 0.0, 1.0) + 0.5 * np.clip(m_margin, 0.0, 1.0)
        # Eşiği geçen bir blok en az 0.5 güven hak eder; sıfırdan başlatmak,
        # eşiğin hemen üstündeki gerçek konuşmayı aşağıdaki katmanlarda eletirdi.
        return float(round(0.5 + 0.5 * score, 3))
