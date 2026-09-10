# ROS'suz Sesli Yanıt Döngüsü (C1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `standalone/track.py`'ye, kafanın takibini bloklamadan çalışan bir sesli yanıt döngüsü eklemek: "hey astro, nasılsın" → sesli cevap.

**Architecture:** Tek mikrofon yakalaması `AudioSource`'ta kalır ve bir halka tampon besler. `VoiceLoop` kendi thread'inde tampondan çeker, `SpeechDetector` ile sözce sınırını bulur, sözceyi `STTRouter`'a, metni `LlmClient`'a, cevabı `TTSRouter`'a verir. `AudioOutputManager.is_playing` + yankı soğuması tek bir `is_robot_speaking` bayrağı üretir; bu bayrak hem gaze'e (kendi sesine dönmesin) hem ses döngüsüne (kendini transkribe etmesin) gider.

**Tech Stack:** Python 3.10, numpy, sounddevice, `openai` (bulut STT/LLM/TTS), paylaşılan ROS'suz kütüphaneler: `astro_audio.stt_router`, `astro_audio.tts_router`, `astro_audio.audio_output_manager`, `astro_audio.speech_detector`, `astro_ai.conversation_session`.

**Spec:** `docs/superpowers/specs/2026-09-04-rossuz-ses-dongusu-design.md`

## Global Constraints

- Testler **repo venv'iyle** koşar: `./.venv/bin/python -m pytest`. Sistem Python'u kullanılmaz.
- Standalone testleri **donanım ve ağ istemez**. Sağlayıcılar (STT/LLM/TTS/çıkış) enjekte edilir; gerçek API çağrısı yalnızca Task 7'nin son adımında, bir kez.
- **Beyin paylaşılır, kopyalanmaz.** Karar veren her şey `astro_audio` / `astro_ai` / `astro_base` içinden import edilir. Yalnız kablolama `standalone/` içinde yaşar. Tek istisna `standalone/llm.py` — `ai_brain_node` içindeki gömülü LLM çağrısının yerine geçen yeni kod.
- Eksik parça programı durdurmaz, görünür kılar: mikrofon/hoparlör/anahtar yoksa tek satır uyarı basılır ve gaze normal çalışır.
- Mevcut 98 standalone testi her task sonunda yeşil kalmalı: `./.venv/bin/python -m pytest standalone/test -q`
- Örnekleme hızı asla varsayılmaz, `sample_rate` ile taşınır.
- Türkçe yorum ve test adları; mevcut dosyaların üslubuna uyulur.

---

### Task 1: Sözce halka tamponu

`AudioSource` bugün konuşma verdisi için 0.6 s'lik kısa bir pencere tutuyor. Sözce için ayrı, daha uzun bir tampon gerekiyor: konuşma bittikten sonra geriye dönüp sözcenin **tamamını** almak lazım.

**Files:**
- Modify: `standalone/sources.py`
- Test: `standalone/test/test_sources.py`

**Interfaces:**
- Consumes: mevcut `AudioSource._observe_speech(mono, timestamp)` çağrı noktaları
- Produces:
  - `AudioSource.read_window(seconds: float) -> Optional[np.ndarray]` (mono, `AudioSource.sample_rate` hızında, en fazla `seconds` kadar, tampon boşsa `None`)
  - `AudioSource.read_since(cursor: int) -> Tuple[np.ndarray, int]` — `cursor`'dan sonra yazılan **yeni** örnekler ve yeni imleç. `cursor=0` tamponun tamamını verir.

> **Neden imleç:** ses döngüsü ana döngüden 30 Hz'te (33 ms) pompalanıyor ama bloklar 64 ms. "Son 64 ms" okumak ardışık çağrılarda örtüşür ve sözceye aynı sesi iki kez yazar; kelimeler kekeleyerek transkribe edilir. İmleç, her örneği tam bir kez teslim eder.

- [ ] **Step 1: Write the failing test**

`standalone/test/test_sources.py` sonuna ekle:

```python
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

        self.assertLessEqual(len(window), int(0.2 * SAMPLE_RATE) + BLOCK_SAMPLES)

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
    "Son 64 ms"i okumak ardisik cagrilarda ortusur ve sozceye ayni sesi iki kez
    yazar; kelimeler kekeleyerek transkribe edilir.
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
```

`standalone/test/test_sources.py` import bloğundaki `from sources import (...)` listesine `UTTERANCE_BUFFER_S` ekle.

`_detached_source()` içine tampon alanlarını ekle (fonksiyonun içinde, `src.sample_rate = SAMPLE_RATE` satırından sonra):

```python
    src._utterance = None
    src._written = 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest standalone/test/test_sources.py -q -k UtteranceBuffer`
Expected: FAIL — `AttributeError: 'AudioSource' object has no attribute 'read_window'`

- [ ] **Step 3: Write minimal implementation**

`standalone/sources.py` sabitlerine ekle (`SPEECH_MAX_AGE_S` satırının altına):

```python
# Sözce tamponu. Konuşma verdisi 0.6 s'lik pencereye bakar; sözcenin kendisi
# bittikten sonra baştan sona geri okunur, o yüzden ayrı ve uzun tutulur.
# 10 s, bir turda söylenebilecek en uzun cümleyi rahatça alır.
UTTERANCE_BUFFER_S = 10.0
```

`AudioSource.__init__` içine, `self._window` satırının yanına:

```python
        self._utterance: Optional[np.ndarray] = None
        self._written: int = 0        # tampona şimdiye kadar yazılan toplam örnek
```

`_observe_speech`'in başına (pencere güncellemesinden önce) ekle:

```python
        self._append_utterance(mono)
```

Ve `latest_speech`'in üstüne iki metot:

```python
    def _append_utterance(self, mono: np.ndarray) -> None:
        """Sözce tamponunu besler. Konuşma penceresinden ayrı tutulur."""
        capacity = int(UTTERANCE_BUFFER_S * self.sample_rate)
        block = np.asarray(mono, dtype=np.float32).ravel()
        with self._lock:
            self._utterance = (block if self._utterance is None
                               else np.concatenate((self._utterance, block)))[-capacity:]
            self._written += len(block)

    def read_since(self, cursor: int):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest standalone/test -q`
Expected: PASS — 106 passed (98 mevcut + 8 yeni)

- [ ] **Step 5: Commit**

```bash
git add standalone/sources.py standalone/test/test_sources.py
git commit -m "feat(audio): sozce icin ayri halka tampon ekle

Konusma verdisi 0.6 s'lik pencereye bakiyor; sozce bittikten sonra
bastan sona geri okunmali. Ayni tamponu paylasirlarsa sozcenin basi
silinir.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: LlmClient

LLM çağrısı `ai_brain_node` içinde 12 ayrı noktada gömülü; yeniden kullanılabilir istemci yok. Tek sorumluluklu bir tane çıkarılıyor.

**Files:**
- Create: `standalone/llm.py`
- Test: `standalone/test/test_llm.py`

**Interfaces:**
- Consumes: yok
- Produces: `LlmClient(client=None, model="gpt-4o-mini", system_prompt=..., timeout_s=20.0)` ve `LlmClient.reply(user_text: str) -> Optional[str]`; `LlmClient.available -> bool`

- [ ] **Step 1: Write the failing test**

Create `standalone/test/test_llm.py`:

```python
#!/usr/bin/env python3
"""LLM istemcisi testleri.

Saglayici enjekte edilir; bu testler ag cagrisi yapmaz. Amac cagrinin
kendisi degil, etrafindaki sozlesme: istemci yoksa program durmaz, bos
cevap yutulur, hata turu dusurur ama dongu yasar.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm import LlmClient  # noqa: E402


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    """OpenAI istemcisinin kullandigimiz tek yuzeyini taklit eder."""

    def __init__(self, content="Iyiyim, sen nasilsin?", raises=None):
        self.content = content
        self.raises = raises
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if outer.raises:
                    raise outer.raises
                return _FakeResponse(outer.content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class LlmClientTests(unittest.TestCase):
    def test_metin_gonderilip_cevap_aliniyor(self):
        fake = _FakeClient(content="Iyiyim.")
        client = LlmClient(client=fake, model="gpt-4o-mini")

        self.assertEqual(client.reply("nasilsin"), "Iyiyim.")

    def test_sistem_istemi_ve_kullanici_metni_iletiliyor(self):
        fake = _FakeClient()
        client = LlmClient(client=fake, system_prompt="Kisa konus.")

        client.reply("merhaba")

        messages = fake.calls[0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Kisa konus.")
        self.assertEqual(messages[-1], {"role": "user", "content": "merhaba"})

    def test_istemci_yoksa_program_durmaz(self):
        """API anahtari olmayan bir masaustunde gaze calismaya devam etmeli."""
        client = LlmClient(client=None)

        self.assertFalse(client.available)
        self.assertIsNone(client.reply("merhaba"))

    def test_saglayici_hatasi_turu_dusurur_ama_yutulur(self):
        client = LlmClient(client=_FakeClient(raises=RuntimeError("429")))

        self.assertIsNone(client.reply("merhaba"))

    def test_bos_cevap_none_olarak_dondurulur(self):
        """Bos metni TTS'e vermek sessiz bir tur ve anlamsiz bir API cagrisidir."""
        client = LlmClient(client=_FakeClient(content="   "))

        self.assertIsNone(client.reply("merhaba"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest standalone/test/test_llm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm'`

- [ ] **Step 3: Write minimal implementation**

Create `standalone/llm.py`:

```python
#!/usr/bin/env python3
"""Tek sorumluluklu LLM istemcisi: metin al, metin döndür.

ROS'lu tarafta bu iş `ai_brain_node` içinde on iki ayrı çağrı noktasına
dağılmış durumda ve her biri kendi hata işlemesini taşıyor. Yeniden
kullanılabilir bir istemci olmadığı için buraya çıkarıldı — bu dosya,
`standalone/` içinde "kablolama değil, yeni kod" olan tek yer.

Sağlayıcı dışarıdan enjekte edilir. Sebebi test değil sadece: anahtarı
olmayan bir masaüstünde program çalışmaya devam etmeli, ve hangi sağlayıcının
konuştuğu çağıranın kararı olmalı.
"""

import os
from typing import Any, List, Optional

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT_S = 20.0

# Kısa tutuluyor: cevap seslendirilecek, okunmayacak. Uzun cevap hem turu
# geciktirir hem de TTS maliyetini büyütür. Persona ve hafıza C2'de gelecek.
DEFAULT_SYSTEM_PROMPT = (
    "Sen ASTRO adında bir sosyal robotsun. Türkçe, kısa ve doğal konuş. "
    "Cevapların en fazla iki cümle olsun."
)


class LlmClient:
    """Bir konuşma turunu metinden metne çevirir."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        temperature: float = 0.55,
    ):
        self._client = client
        self.model = model
        self.system_prompt = system_prompt
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def reply(self, user_text: str) -> Optional[str]:
        """Bir tur. Cevap üretilemezse None — çağıran turu atlar, döngü yaşar."""
        if self._client is None or not user_text or not user_text.strip():
            return None

        messages: List[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text.strip()},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                timeout=self.timeout_s,
            )
        except Exception as exc:
            # Sağlayıcı hatası bir turu düşürür, programı değil: kota, ağ ve
            # hız sınırı hataları konuşma sırasında olağan.
            self.last_error = str(exc)
            return None

        try:
            text = (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError) as exc:
            self.last_error = f"beklenmeyen cevap bicimi: {exc}"
            return None

        return text or None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest standalone/test -q`
Expected: PASS — 111 passed

- [ ] **Step 5: Commit**

```bash
git add standalone/llm.py standalone/test/test_llm.py
git commit -m "feat(voice): ai_brain_node icindeki gomulu LLM cagrisini istemciye cikar

12 ayri cagri noktasina dagilmis, yeniden kullanilamiyordu. Saglayici
enjekte edilebilir: anahtari olmayan masaustunde program calismaya devam
eder ve testler ag cagrisi yapmaz.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `is_robot_speaking` gaze'e ulaşsın

`audio_perception.process_raw_doa` bu parametreyi zaten kabul ediyor ve güveni 0.15 ile çarpıyor. Standalone hiç geçmiyor — yani robot konuşunca kendi sesine döner.

**Files:**
- Modify: `standalone/tracker.py`
- Test: `standalone/test/test_tracker.py`

**Interfaces:**
- Consumes: Task 2'den bir şey yok
- Produces: `GazeTracker.step(..., is_robot_speaking: bool = False)`

- [ ] **Step 1: Write the failing test**

`standalone/test/test_tracker.py` içine, `if __name__` bloğunun ÜSTÜNE ekle:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest standalone/test/test_tracker.py -q -k OwnVoice`
Expected: FAIL — `TypeError: GazeTracker.step() got an unexpected keyword argument 'is_robot_speaking'`

- [ ] **Step 3: Write minimal implementation**

`standalone/tracker.py` — `step` imzasına parametre ekle (`speech=None` satırından sonra):

```python
        is_robot_speaking: bool = False,
```

Aynı metodun gövdesinde ses alma satırını değiştir:

```python
        if doa_deg is not None and speech is not None and speech.is_speech:
            self._ingest_audio(doa_deg, timestamp, float(speech.confidence),
                               is_robot_speaking)
```

`_ingest_audio` imzasını ve çağrısını güncelle:

```python
    def _ingest_audio(self, doa_deg: float, timestamp: float, confidence: float,
                      is_robot_speaking: bool = False) -> None:
```

ve gövdesindeki `process_raw_doa` çağrısına ekle:

```python
            is_robot_speaking=is_robot_speaking,
```

`step`'in docstring'ine ekle:

```
        `is_robot_speaking` hoparlör çalarken True olur. Hoparlör mikrofonun
        yanında; robot konuştuğu anda güçlü ve harmonik bir kerteriz üretilir ve
        o kerteriz her zaman hoparlörü gösterir. Konuşma filtresi bunu elemez —
        robotun sesi de konuşmadır.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest standalone/test -q`
Expected: PASS — 113 passed

- [ ] **Step 5: Commit**

```bash
git add standalone/tracker.py standalone/test/test_tracker.py
git commit -m "fix(gaze): robot konusurken kendi sesine donmesin

process_raw_doa is_robot_speaking parametresini zaten kabul ediyordu ve
guveni 0.15 ile carpiyor; standalone hic gecirmiyordu. Konusma filtresi
bunu elemez, cunku robotun sesi de konusmadir.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Sözce sınırı

`VoiceLoop`'un ilk yarısı: konuşma nerede başladı, nerede bitti. STT/LLM/TTS henüz yok.

**Files:**
- Create: `standalone/voice.py`
- Test: `standalone/test/test_voice.py`

**Interfaces:**
- Consumes: `AudioSource.read_window` (Task 1)
- Produces: `UtteranceTracker(sample_rate, silence_s=0.8, max_s=10.0)` ve `UtteranceTracker.feed(is_speech: bool, block: np.ndarray, timestamp: float) -> Optional[np.ndarray]` — sözce kapandığı çevrimde sözcenin tamamını döndürür, aksi halde `None`

- [ ] **Step 1: Write the failing test**

Create `standalone/test/test_voice.py`:

```python
#!/usr/bin/env python3
"""Sesli yanit dongusu testleri. Donanim ve ag istemez.

Sozce siniri burada test ediliyor: konusma nerede basladi, nerede bitti.
Cumle ici duraklama sozceyi kapatmamali (hece arasi <=0.25 sn, virgul
duraklamasi ~0.5 sn), sozce sonu kapatmali.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_path  # noqa: F401,E402
from voice import UtteranceTracker  # noqa: E402

SAMPLE_RATE = 16000
BLOCK = 1024
BLOCK_S = BLOCK / SAMPLE_RATE      # 64 ms


def _block(value=0.2):
    return np.full(BLOCK, value, dtype=np.float32)


class UtteranceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tracker = UtteranceTracker(sample_rate=SAMPLE_RATE, silence_s=0.8)

    def _feed(self, pattern, t0=0.0):
        """pattern: her elemani bir blok icin is_speech. Kapanan sozceyi dondurur."""
        closed = []
        t = t0
        for is_speech in pattern:
            out = self.tracker.feed(is_speech, _block(), timestamp=t)
            if out is not None:
                closed.append(out)
            t += BLOCK_S
        return closed

    def test_sessizlik_sozce_acmaz(self):
        self.assertEqual(self._feed([False] * 30), [])

    def test_konusma_sonrasi_yeterli_sessizlik_sozceyi_kapatir(self):
        # 10 blok konusma (0.64 sn), sonra 16 blok sessizlik (1.02 sn > 0.8)
        closed = self._feed([True] * 10 + [False] * 16)

        self.assertEqual(len(closed), 1, "sozce kapanmadi")
        self.assertGreaterEqual(len(closed[0]), 10 * BLOCK,
                                "sozce konusma bloklarinin tamamini icermiyor")

    def test_cumle_ici_duraklama_sozceyi_bolmez(self):
        """Virgul duraklamasi ~0.5 sn; 0.8 sn esigi bunu gecirmemeli."""
        # 6 blok konusma, 7 blok sessizlik (0.45 sn), 6 blok konusma, sonra kapanis
        closed = self._feed([True] * 6 + [False] * 7 + [True] * 6 + [False] * 16)

        self.assertEqual(len(closed), 1,
                         f"cumle ici duraklama sozceyi boldu: {len(closed)} parca")

    def test_sozce_kapanmadan_once_hicbir_sey_dondurulmez(self):
        self.assertEqual(self._feed([True] * 10 + [False] * 5), [])

    def test_cok_uzun_sozce_ust_sinirda_kapanir(self):
        """Susmayan bir kaynak tamponu sonsuza kadar buyutmemeli."""
        tracker = UtteranceTracker(sample_rate=SAMPLE_RATE, silence_s=0.8, max_s=1.0)
        closed = []
        t = 0.0
        for _ in range(40):                        # 2.5 sn kesintisiz konusma
            out = tracker.feed(True, _block(), timestamp=t)
            if out is not None:
                closed.append(out)
            t += BLOCK_S

        self.assertGreaterEqual(len(closed), 1, "ust sinir uygulanmadi")
        self.assertLessEqual(len(closed[0]), int(1.0 * SAMPLE_RATE) + BLOCK)

    def test_kapanan_sozce_sonraki_sozceye_karismaz(self):
        first = self._feed([True] * 8 + [False] * 16)
        second = self._feed([True] * 8 + [False] * 16, t0=100.0)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertLessEqual(len(second[0]), 10 * BLOCK,
                             "ikinci sozce birincinin sesini tasiyor")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest standalone/test/test_voice.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'voice'`

- [ ] **Step 3: Write minimal implementation**

Create `standalone/voice.py`:

```python
#!/usr/bin/env python3
"""Sesli yanıt döngüsü — ROS'suz.

ROS'lu tarafta bu iş beş düğüme dağılmış ve aralarındaki her şey topic:
audio_stream_node → speech_recognition_node → ai_brain_node → tts_node, artı
yankı bastırma için geri dönen `/tts/speaking`. Buradaki dosya o topolojinin
yerine geçen kablolama; karar veren nesnelerin hepsi paylaşılan kütüphanelerden
geliyor (`STTRouter`, `TTSRouter`, `AudioOutputManager`, `ConversationSession`).
"""

from typing import Optional

import numpy as np

# Sözce sonu eşiği. Cümle içi duraklamayı sözce sonundan ayırmalı: hece arası
# ≤0.25 s, virgül duraklaması ~0.5 s, cümle sonu daha uzun. 0.8 s ikisinin
# arasında ve tur gecikmesini hissedilir kılmıyor. Gerçek kayıtla yeniden
# ayarlanacak, o yüzden dışarıdan verilebiliyor.
DEFAULT_SILENCE_S = 0.8

# Susmayan bir kaynak (açık televizyon, uğultu) tamponu sonsuza büyütmesin.
DEFAULT_MAX_UTTERANCE_S = 10.0


class UtteranceTracker:
    """Blok blok gelen konuşma kararlarından sözce sınırı çıkarır.

    Kararın kendisini üretmez — o `astro_audio.speech_detector`'ın işi. Burada
    yalnızca "ne zaman başladı, ne zaman bitti" var, çünkü sözce sınırı bir
    zamanlama sorusu, akustik bir soru değil.
    """

    def __init__(
        self,
        sample_rate: int,
        silence_s: float = DEFAULT_SILENCE_S,
        max_s: float = DEFAULT_MAX_UTTERANCE_S,
    ):
        self.sample_rate = int(sample_rate)
        self.silence_s = float(silence_s)
        self.max_s = float(max_s)
        self._chunks: list = []
        self._silence_started_at: Optional[float] = None
        self._samples = 0

    @property
    def active(self) -> bool:
        return bool(self._chunks)

    def feed(self, is_speech: bool, block: np.ndarray, timestamp: float) -> Optional[np.ndarray]:
        """Bir bloğu işler. Sözce bu çevrimde kapandıysa tamamını döndürür."""
        samples = np.asarray(block, dtype=np.float32).ravel()

        if is_speech:
            self._chunks.append(samples)
            self._samples += len(samples)
            self._silence_started_at = None
            if self._samples >= int(self.max_s * self.sample_rate):
                return self._close()
            return None

        if not self._chunks:
            # Sözce başlamadı: sessizlik biriktirilmez.
            return None

        # Sözce açıkken gelen sessizlik saklanır — cümle içi duraklama sözcenin
        # parçasıdır ve atılırsa kelimeler birbirine yapışır.
        self._chunks.append(samples)
        self._samples += len(samples)

        if self._silence_started_at is None:
            self._silence_started_at = timestamp
        elif (timestamp - self._silence_started_at) >= self.silence_s:
            return self._close()
        return None

    def _close(self) -> np.ndarray:
        utterance = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        self._chunks = []
        self._samples = 0
        self._silence_started_at = None
        return utterance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest standalone/test -q`
Expected: PASS — 119 passed

- [ ] **Step 5: Commit**

```bash
git add standalone/voice.py standalone/test/test_voice.py
git commit -m "feat(voice): sozce siniri (baslangic/bitis) ekle

Sozce siniri zamanlama sorusu; akustik karar speech_detector'da kaliyor.
Esik 0.8 sn: cumle ici duraklamayi (virgul ~0.5 sn) sozce sonundan ayirir.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Tur — STT → wake word → LLM → TTS

**Files:**
- Modify: `standalone/voice.py`
- Modify: `standalone/core_path.py` (astro_ai yola eklenir — ön uçuş bulgusu)
- Test: `standalone/test/test_voice.py`

**Interfaces:**
- Consumes: `UtteranceTracker` (Task 4), `LlmClient.reply` (Task 2)
- Produces: `VoiceLoop(audio, stt=None, llm=None, tts=None, output=None, session=None, wake_word="hey astro", speech=None, silence_s=0.8)`; `VoiceLoop.handle_utterance(utterance: np.ndarray, sample_rate: int) -> Optional[str]` (söylenen cevap ya da `None`); `VoiceLoop.is_speaking -> bool`

- [ ] **Step 1: Write the failing test**

`standalone/test/test_voice.py` içine, `if __name__` bloğunun ÜSTÜNE ekle:

```python
class _FakeStt:
    def __init__(self, text="hey astro nasilsin"):
        self.text = text
        self.calls = []

    def transcribe(self, audio_arr, wav_bytes, sample_rate=16000):
        self.calls.append((len(audio_arr), sample_rate))

        class _Result:
            pass

        result = _Result()
        result.text = self.text
        result.provider = "fake"
        return result


class _FakeLlm:
    def __init__(self, answer="Iyiyim."):
        self.answer = answer
        self.prompts = []
        self.available = True

    def reply(self, user_text):
        self.prompts.append(user_text)
        return self.answer


class _FakeTts:
    def __init__(self):
        self.spoken = []

    def synthesize_and_play(self, text, generation_id, output_manager=None,
                            language="tr", realtime_fallback_reason=None):
        self.spoken.append(text)
        return None


class _FakeOutput:
    def __init__(self):
        self._gen = 0
        self.is_playing = False
        self.interrupts = 0

    def new_generation(self):
        self._gen += 1
        return self._gen

    def interrupt(self, new_generation_id=None):
        self.interrupts += 1
        self._gen += 1
        return self._gen


def _voice(stt=None, llm=None, tts=None, output=None, **kwargs):
    from voice import VoiceLoop

    return VoiceLoop(audio=None, stt=stt or _FakeStt(), llm=llm or _FakeLlm(),
                     tts=tts or _FakeTts(), output=output or _FakeOutput(), **kwargs)


class TurnTests(unittest.TestCase):
    UTTERANCE = np.full(16000, 0.2, dtype=np.float32)

    def test_wake_word_ile_oturum_acilir_ve_cevap_seslendirilir(self):
        tts = _FakeTts()
        loop = _voice(stt=_FakeStt("hey astro nasilsin"), tts=tts)

        said = loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertEqual(said, "Iyiyim.")
        self.assertEqual(tts.spoken, ["Iyiyim."])

    def test_oturum_kapaliyken_wake_wordsuz_tur_dusurulur(self):
        tts = _FakeTts()
        loop = _voice(stt=_FakeStt("bugun hava nasil"), tts=tts)

        said = loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertIsNone(said)
        self.assertEqual(tts.spoken, [], "oturum kapaliyken konustu")

    def test_oturum_acikken_wake_word_gerekmez(self):
        tts = _FakeTts()
        loop = _voice(stt=_FakeStt("hey astro merhaba"), tts=tts)
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        loop.stt = _FakeStt("bugun hava nasil")
        said = loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertIsNotNone(said)
        self.assertEqual(len(tts.spoken), 2)

    def test_wake_word_metinden_temizlenip_llme_gider(self):
        llm = _FakeLlm()
        loop = _voice(stt=_FakeStt("hey astro bugun hava nasil"), llm=llm)

        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertNotIn("astro", llm.prompts[0].lower(),
                         f"wake word temizlenmemis: {llm.prompts[0]!r}")

    def test_bos_transkript_tur_acmaz(self):
        tts = _FakeTts()
        loop = _voice(stt=_FakeStt(""), tts=tts)

        self.assertIsNone(loop.handle_utterance(self.UTTERANCE, sample_rate=16000))
        self.assertEqual(tts.spoken, [])

    def test_llm_cevap_veremezse_sessiz_kalinir(self):
        tts = _FakeTts()
        loop = _voice(stt=_FakeStt("hey astro selam"), llm=_FakeLlm(answer=None), tts=tts)

        self.assertIsNone(loop.handle_utterance(self.UTTERANCE, sample_rate=16000))
        self.assertEqual(tts.spoken, [], "LLM sussa da TTS konustu")

    def test_stt_16_khze_donusturulmus_ses_gorur(self):
        """Tampon 44.1 kHz olabilir; STT 16 kHz bekler. Donusum tek noktada."""
        stt = _FakeStt("hey astro selam")
        loop = _voice(stt=stt)
        utterance = np.full(44100, 0.2, dtype=np.float32)     # 1 sn @44.1 kHz

        loop.handle_utterance(utterance, sample_rate=44100)

        length, rate = stt.calls[0]
        self.assertEqual(rate, 16000)
        self.assertAlmostEqual(length, 16000, delta=100,
                               msg="ses STT'ye yakalama hizinda verilmis")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest standalone/test/test_voice.py -q -k Turn`
Expected: FAIL — `ImportError: cannot import name 'VoiceLoop' from 'voice'`

- [ ] **Step 3: Write minimal implementation**

**Önce `standalone/core_path.py`'yi düzelt.** Bugün yalnızca `astro_base`,
`astro_vision` ve `astro_audio`'yu yola ekliyor; `astro_ai` yok, ve aşağıdaki
import onsuz `ModuleNotFoundError` ile düşer (ön uçuş taramasında doğrulandı).
`_PACKAGES` demetine ekle:

```python
    os.path.join(REPO, "ros2_ws", "src", "astro_ai"),
```

`core_path.py`'nin docstring'inin son paragrafını da güncelle:

```
So astro_base/gaze, the astro_vision helpers and the astro_ai conversation
pieces are imported from where they live.
```

Sonra `standalone/voice.py` başındaki import bloğunu değiştir:

```python
import io
import wave
from typing import Any, Optional

import numpy as np

import core_path  # noqa: F401
from astro_ai.conversation_session import ConversationSession  # noqa: E402
from astro_audio.speech_detector import SpeechDetector  # noqa: E402
```

Sabitlerin altına ekle:

```python
# STT'nin beklediği hız. Tampon yakalama hızında tutulur (dizi modunda 16 kHz,
# stereo modda aygıtın kendi hızı — laptopta 44.1 kHz) ve dönüşüm tek noktada,
# STT'ye verilmeden hemen önce yapılır. Hızın sessizce varsayılması bu depoda
# daha önce bedel ödetti: int16 eşiği float32 akışa uygulanmıştı ve kapı her
# bloğu eliyordu.
STT_SAMPLE_RATE = 16000

WAKE_WORD = "hey astro"
```

Dosya sonuna ekle:

```python
def resample_to(mono: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Doğrusal ara değerle yeniden örnekler.

    Konuşma tanıma için yeterli: STT modelleri kendi ön işlemesinde zaten
    bant sınırlıyor, ve buradaki tek amaç hızı sözleşmeye getirmek.
    """
    if from_rate == to_rate or len(mono) == 0:
        return np.asarray(mono, dtype=np.float32)
    count = int(round(len(mono) * (to_rate / float(from_rate))))
    if count <= 0:
        return np.zeros(0, dtype=np.float32)
    source = np.linspace(0.0, len(mono) - 1, num=count)
    return np.interp(source, np.arange(len(mono)), mono).astype(np.float32)


def to_wav_bytes(mono: np.ndarray, sample_rate: int) -> bytes:
    """float32 [-1, 1] mono sesi 16-bit PCM WAV'a çevirir (STT'nin istediği)."""
    clipped = np.clip(np.asarray(mono, dtype=np.float32), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm16.tobytes())
    return buffer.getvalue()


class VoiceLoop:
    """Bir konuşma turunu baştan sona yürütür.

    Bütün sağlayıcılar enjekte edilebilir: testler sahtelerle koşar ve ağ
    çağrısı yapmaz, ve hangi sağlayıcının konuştuğu çağıranın kararı kalır.
    """

    def __init__(
        self,
        audio: Any = None,
        stt: Any = None,
        llm: Any = None,
        tts: Any = None,
        output: Any = None,
        session: Optional[ConversationSession] = None,
        speech: Optional[SpeechDetector] = None,
        wake_word: str = WAKE_WORD,
        silence_s: float = DEFAULT_SILENCE_S,
    ):
        self.audio = audio
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.output = output
        self.session = session or ConversationSession()
        self.speech = speech or SpeechDetector(sample_rate=STT_SAMPLE_RATE)
        self.wake_word = wake_word
        self.silence_s = float(silence_s)

    @property
    def is_speaking(self) -> bool:
        return bool(self.output is not None and getattr(self.output, "is_playing", False))

    def handle_utterance(self, utterance: np.ndarray, sample_rate: int) -> Optional[str]:
        """Bir sözceyi tura çevirir. Söylenen cevabı, yoksa None döndürür."""
        if self.stt is None or utterance is None or len(utterance) == 0:
            return None

        audio = resample_to(utterance, sample_rate, STT_SAMPLE_RATE)
        result = self.stt.transcribe(audio, to_wav_bytes(audio, STT_SAMPLE_RATE),
                                     sample_rate=STT_SAMPLE_RATE)
        text = (getattr(result, "text", None) or "").strip()
        if not text:
            return None

        has_wake, clean = self.session.is_wake_word(text, self.wake_word)
        if has_wake:
            self.session.activate_session(reason="wake_word")
        elif not self.session.is_active():
            # Oturum kapalı ve çağrılmadık: bu konuşma bize değil.
            return None
        else:
            clean = text

        self.session.record_user_speech()
        prompt = clean.strip() or text
        answer = self.llm.reply(prompt) if self.llm is not None else None
        if not answer:
            return None

        self._speak(answer)
        return answer

    def _speak(self, text: str) -> None:
        if self.tts is None:
            return
        generation_id = self.output.new_generation() if self.output is not None else 0
        self.tts.synthesize_and_play(text, generation_id=generation_id,
                                     output_manager=self.output, language="tr")
        self.session.record_robot_speech()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest standalone/test -q`
Expected: PASS — 126 passed

- [ ] **Step 5: Commit**

```bash
git add standalone/voice.py standalone/test/test_voice.py
git commit -m "feat(voice): sozce -> STT -> wake word -> LLM -> TTS turu

Karar veren nesnelerin hepsi paylasilan kutuphanelerden: STTRouter,
TTSRouter, ConversationSession. Ornekleme hizi tek noktada, STT'ye
verilmeden hemen once donusturuluyor.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Yankı bastırma ve barge-in

**Files:**
- Modify: `standalone/voice.py`
- Test: `standalone/test/test_voice.py`

**Interfaces:**
- Consumes: `VoiceLoop` (Task 5)
- Produces: `VoiceLoop.note_playback(is_playing: bool, timestamp: float)`; `VoiceLoop.is_speaking_at(timestamp: float) -> bool`; `VoiceLoop.on_block(is_speech: bool, block: np.ndarray, timestamp: float) -> Optional[np.ndarray]`

- [ ] **Step 1: Write the failing test**

`standalone/test/test_voice.py` içine, `if __name__` bloğunun ÜSTÜNE ekle:

```python
class EchoAndBargeInTests(unittest.TestCase):
    """Robot kendini duymamali, ama kullanici araya girince susmali."""

    def _loop(self, output=None):
        return _voice(output=output or _FakeOutput(), echo_cooldown_s=0.65)

    def test_hoparlor_sustuktan_sonra_sogumada_hala_konusuyor_sayilir(self):
        """Bayrak aninda duserse mikrofona yolda olan kendi sesi transkribe edilir."""
        loop = self._loop()

        loop.note_playback(True, timestamp=10.0)
        loop.note_playback(False, timestamp=11.0)

        self.assertTrue(loop.is_speaking_at(11.3), "soguma penceresi yok")
        self.assertFalse(loop.is_speaking_at(11.8), "soguma hic bitmiyor")

    def test_robot_konusurken_gelen_konusma_transkribe_edilmez(self):
        loop = self._loop()
        loop.note_playback(True, timestamp=10.0)

        closed = []
        t = 10.0
        for is_speech in [True] * 10 + [False] * 16:
            out = loop.on_block(is_speech, _block(), timestamp=t)
            if out is not None:
                closed.append(out)
            t += BLOCK_S

        self.assertEqual(closed, [], "robot konusurken sozce transkripsiyona gitti")

    def test_robot_konusurken_gercek_konusma_calmayi_keser(self):
        output = _FakeOutput()
        loop = self._loop(output=output)
        loop.note_playback(True, timestamp=10.0)

        t = 10.0
        for _ in range(6):                       # 0.38 sn kesintisiz konusma
            loop.on_block(True, _block(), timestamp=t)
            t += BLOCK_S

        self.assertGreaterEqual(output.interrupts, 1, "barge-in calismadi")

    def test_robot_konusurken_gurultu_calmayi_kesmez(self):
        output = _FakeOutput()
        loop = self._loop(output=output)
        loop.note_playback(True, timestamp=10.0)

        t = 10.0
        for _ in range(30):
            loop.on_block(False, _block(), timestamp=t)
            t += BLOCK_S

        self.assertEqual(output.interrupts, 0, "gurultu calmayi kesti")

    def test_robot_susarken_sozce_normal_kapanir(self):
        loop = self._loop()

        closed = []
        t = 0.0
        for is_speech in [True] * 10 + [False] * 16:
            out = loop.on_block(is_speech, _block(), timestamp=t)
            if out is not None:
                closed.append(out)
            t += BLOCK_S

        self.assertEqual(len(closed), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest standalone/test/test_voice.py -q -k EchoAndBargeIn`
Expected: FAIL — `TypeError: VoiceLoop.__init__() got an unexpected keyword argument 'echo_cooldown_s'`

- [ ] **Step 3: Write minimal implementation**

`standalone/voice.py` sabitlerine ekle:

```python
# Hoparlör sustuktan sonra mikrofonun kendi sesini duymayı bırakması zaman alır.
# .env'deki ECHO_MUTE_COOLDOWN_S ile aynı sayı; bayrak anında düşerse yolda olan
# kendi ses transkribe edilir ve robot kendine cevap verir.
DEFAULT_ECHO_COOLDOWN_S = 0.65

# Robot konuşurken araya girmeyi ciddiye almadan önce görülmesi gereken kesintisiz
# konuşma. Tek blok (64 ms) hoparlör sızıntısıyla da olabilir; 0.3 s bir hecedir.
DEFAULT_BARGE_IN_S = 0.3
```

`VoiceLoop.__init__` imzasına ekle (`silence_s` satırından sonra):

```python
        echo_cooldown_s: float = DEFAULT_ECHO_COOLDOWN_S,
        barge_in_s: float = DEFAULT_BARGE_IN_S,
```

ve gövdesinin sonuna:

```python
        self.echo_cooldown_s = float(echo_cooldown_s)
        self.barge_in_s = float(barge_in_s)
        self._playing = False
        self._playback_ended_at: Optional[float] = None
        self._barge_in_since: Optional[float] = None
        # Sözce takipçisi **yakalama** hızıyla kurulur, STT hızıyla değil. Bloklar
        # tampondan yakalama hızında geliyor (dizi modunda 16 kHz, stereo modda
        # aygıtın kendi hızı — laptopta 44.1 kHz); STT hızını varsaymak sözce üst
        # sınırını 2.75 kat yanlış hesaplardı. Dönüşüm yalnızca STT'ye girerken.
        capture_rate = int(getattr(audio, "sample_rate", 0) or STT_SAMPLE_RATE)
        self._utterance = UtteranceTracker(sample_rate=capture_rate,
                                           silence_s=self.silence_s)
```

`is_speaking` property'sini değiştir ve iki metot ekle:

```python
    @property
    def is_speaking(self) -> bool:
        """Anlık durum. Zamana bağlı karar için `is_speaking_at` kullanılır."""
        if self.output is not None and getattr(self.output, "is_playing", False):
            return True
        return self._playing

    def note_playback(self, is_playing: bool, timestamp: float) -> None:
        """`AudioOutputManager`'ın çalma durumu değişince çağrılır."""
        if not is_playing and self._playing:
            self._playback_ended_at = float(timestamp)
        self._playing = bool(is_playing)

    def is_speaking_at(self, timestamp: float) -> bool:
        """Robot şu anda konuşuyor mu — yankı soğuması dahil."""
        if self._playing:
            return True
        if self._playback_ended_at is None:
            return False
        return (timestamp - self._playback_ended_at) < self.echo_cooldown_s

    def on_block(self, is_speech: bool, block: np.ndarray,
                 timestamp: float) -> Optional[np.ndarray]:
        """Bir ses bloğunu işler. Sözce kapandıysa tamamını döndürür.

        Robot konuşurken sözce biriktirilmez — biriktirilse robotun kendi sesi
        transkribe edilirdi. Blok yine de okunur, çünkü araya girmeyi ancak
        dinleyerek fark edebiliriz.
        """
        if self.is_speaking_at(timestamp):
            self._check_barge_in(is_speech, timestamp)
            return None

        self._barge_in_since = None
        return self._utterance.feed(is_speech, block, timestamp)

    def _check_barge_in(self, is_speech: bool, timestamp: float) -> None:
        if not is_speech:
            self._barge_in_since = None
            return
        if self._barge_in_since is None:
            self._barge_in_since = timestamp
            return
        if (timestamp - self._barge_in_since) >= self.barge_in_s:
            if self.output is not None:
                self.output.interrupt()
            self._playing = False
            self._playback_ended_at = timestamp
            self._barge_in_since = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest standalone/test -q`
Expected: PASS — 131 passed

- [ ] **Step 5: Commit**

```bash
git add standalone/voice.py standalone/test/test_voice.py
git commit -m "feat(voice): yanki bastirma ve barge-in

Hoparlor sustuktan sonra 0.65 sn soguma: bayrak aninda duserse mikrofona
yolda olan kendi ses transkribe edilir. Barge-in 0.3 sn kesintisiz konusma
ister; tek blok hoparlor sizintisiyla da olabilir.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: track.py'ye bağla, bağımlılıkları kur, gerçek tur

**Files:**
- Modify: `standalone/track.py`
- Modify: `standalone/voice.py` (fabrika fonksiyonu)
- Modify: `standalone/README.md`
- Test: `standalone/test/test_voice.py`

**Interfaces:**
- Consumes: hepsi
- Produces: `voice.build_default_loop(audio) -> Optional[VoiceLoop]` — sağlayıcılar kurulabiliyorsa döngü, kurulamıyorsa `None` (sebep `voice.LAST_SETUP_ERROR`'da)

- [ ] **Step 1: Write the failing test**

`standalone/test/test_voice.py` içine, `if __name__` bloğunun ÜSTÜNE ekle:

```python
class DegradationTests(unittest.TestCase):
    """Eksik parca programi durdurmaz, gorunur kilar."""

    def test_openai_anahtari_yoksa_dongu_kurulmaz_ama_sebep_soylenir(self):
        import voice

        loop = voice.build_default_loop(audio=None, api_key="")

        self.assertIsNone(loop)
        self.assertIn("anahtar", voice.LAST_SETUP_ERROR.lower())

    def test_calma_bildirimi_donguye_baglanir(self):
        """Baglanmazsa yanki bastirma ve barge-in uretimde sessizce olu kalir.

        `is_speaking_at` yalnizca `note_playback` ile hareket eder; cikis
        yoneticisi geri cagriyi cagirmazsa robot kendi sesini transkribe eder.
        """
        import inspect
        import voice

        source = inspect.getsource(voice.build_default_loop)

        self.assertIn("on_playback_state_change", source,
                      "AudioOutputManager calma durumu geri cagrisiyla kurulmuyor")
        self.assertIn("note_playback", source,
                      "geri cagri VoiceLoop.note_playback'e baglanmiyor")

    def test_llm_yoksa_tur_sessizce_atlanir(self):
        loop = _voice(llm=None)

        said = loop.handle_utterance(np.full(16000, 0.2, np.float32), sample_rate=16000)

        self.assertIsNone(said)

    def test_hoparlor_yoksa_stt_ve_llm_yine_calisir(self):
        """Cikis olmadan da tur islenmeli; yalniz seslendirme dusmeli."""
        llm = _FakeLlm()
        loop = _voice(stt=_FakeStt("hey astro selam"), llm=llm, tts=None)

        said = loop.handle_utterance(np.full(16000, 0.2, np.float32), sample_rate=16000)

        self.assertEqual(said, "Iyiyim.")
        self.assertEqual(len(llm.prompts), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python -m pytest standalone/test/test_voice.py -q -k Degradation`
Expected: FAIL — `AttributeError: module 'voice' has no attribute 'build_default_loop'`

- [ ] **Step 3: Install dependencies and write the implementation**

Bağımlılıklar `requirements.in`'de zaten tanımlı, yalnızca bu venv'e kurulmamış:

```bash
./.venv/bin/pip install openai python-dotenv
```

`standalone/voice.py` sonuna ekle:

```python
LAST_SETUP_ERROR = ""


def build_default_loop(audio, api_key: Optional[str] = None, **kwargs):
    """Bulut sağlayıcılarla bir döngü kurar; kuramazsa None döner ve sebebi yazar.

    Anahtarı olmayan bir masaüstünde `track.py` çalışmaya devam etmeli — takip
    sesli yanıta bağımlı değil. Bu yüzden burada istisna fırlatılmıyor.
    """
    global LAST_SETUP_ERROR

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        LAST_SETUP_ERROR = "OPENAI_API_KEY yok — sesli yanıt kapalı, takip çalışmaya devam ediyor"
        return None

    try:
        from openai import OpenAI

        from astro_audio.audio_output_manager import AudioOutputManager
        from astro_audio.openai_tts_engine import OpenAITTSEngine
        from astro_audio.stt_router import STTRouter
        from astro_audio.tts_router import TTSRouter
    except ImportError as exc:
        LAST_SETUP_ERROR = f"paket eksik ({exc}) — `./.venv/bin/pip install openai python-dotenv`"
        return None

    # `note_playback` bağlanmazsa yankı bastırma ve barge-in üretimde sessizce ölü
    # kalır: `is_speaking_at` yalnızca bu bildirimle hareket eder. Çıkış yöneticisi
    # döngüden önce kurulmak zorunda olduğu için geri çağrı, döngüyü sonradan
    # dolduran bir hücreden okur.
    cell = {}

    def _on_playback(is_playing: bool) -> None:
        loop = cell.get("loop")
        if loop is not None:
            loop.note_playback(bool(is_playing), time.monotonic())

    try:
        client = OpenAI(api_key=api_key)
        output = AudioOutputManager(on_playback_state_change=_on_playback)
        tts = TTSRouter(openai_tts_engine=OpenAITTSEngine(client=client),
                        edge_tts_enabled=False, output_manager=output)
        stt = STTRouter(openai_client=client)
    except Exception as exc:
        LAST_SETUP_ERROR = f"sağlayıcılar kurulamadı: {exc}"
        return None

    LAST_SETUP_ERROR = ""
    loop = VoiceLoop(audio=audio, stt=stt, llm=LlmClient(client=client),
                     tts=tts, output=output, **kwargs)
    cell["loop"] = loop
    return loop
```

Ve `standalone/voice.py` import bloğuna `import time` ekle (yoksa).

`standalone/voice.py` import bloğuna ekle:

```python
import os

from llm import LlmClient  # noqa: E402
```

`standalone/track.py` — `--no-voice` seçeneği ekle (`--no-window` satırının yanına):

```python
    parser.add_argument("--no-voice", action="store_true",
                        help="Sesli yanıtı kapat (yalnızca takip)")
```

`audio.start()` çağrısından sonra:

```python
    import voice as voice_module

    voice_loop = None
    if not opts.no_voice and audio.available:
        voice_loop = voice_module.build_default_loop(audio)
        if voice_loop is None:
            print(f"🗣️  {voice_module.LAST_SETUP_ERROR}")
        else:
            print(f"🗣️  Sesli yanıt açık — uyandırma sözcüğü: '{voice_loop.wake_word}'")
```

Ana döngüde, `speech = audio.latest_speech(now)` satırından sonra:

```python
            if voice_loop is not None:
                voice_loop.pump(now)
```

`tracker.step` çağrısına ekle:

```python
                is_robot_speaking=voice_loop.is_speaking_at(now) if voice_loop else False,
```

`finally` bloğunda `audio.close()` satırından önce:

```python
        if voice_loop is not None:
            voice_loop.stop()
```

`VoiceLoop`'a çalışma zamanı metotlarını ekle (`_check_barge_in`'in altına):

```python
    def pump(self, timestamp: float) -> None:
        """Ana döngüden her karede çağrılır; iş varsa arka plana atar.

        Kafa 30 Hz'te akıyor ve bir LLM turu saniyeler sürüyor — tur bu thread'de
        koşarsa takip donar. O yüzden burada yalnızca sözce toplanır; tur kapanınca
        işi bir arka plan thread'i alır.
        """
        if self.audio is None or self._turn_running:
            return

        block, self._cursor = self.audio.read_since(self._cursor)
        if len(block) == 0:
            return

        verdict = self.audio.latest_speech(timestamp)
        is_speech = bool(verdict is not None and verdict.is_speech)
        utterance = self.on_block(is_speech, block, timestamp)
        if utterance is None:
            return

        self._turn_running = True
        thread = threading.Thread(
            target=self._run_turn,
            args=(utterance, int(self.audio.sample_rate)),
            daemon=True,
        )
        thread.start()
        self._thread = thread

    def _run_turn(self, utterance, sample_rate: int) -> None:
        try:
            self.handle_utterance(utterance, sample_rate)
        finally:
            self._turn_running = False

    def stop(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
```

`VoiceLoop.__init__` gövdesinin sonuna:

```python
        self._turn_running = False
        self._thread = None
        # Halka tampon imleci: her örnek tam bir kez teslim edilsin.
        self._cursor = 0
```

ve `standalone/voice.py` import bloğuna `import threading` ekle.

> **Tur koşarken gelen ses:** `_turn_running` doğruyken `pump` erken dönüyor, yani
> imleç ilerlemiyor ve o sırada konuşulanlar tamponda birikiyor. Tur bitince
> imleçten devam edilir; 10 s'lik tamponu aşan kısım düşer. Bu bilinçli: tur
> sırasındaki sesi biriktirip sonra transkribe etmek, robotun bir önceki soruya
> cevap verirken duyduğu her şeyi sıraya sokması demek olurdu.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/bin/python -m pytest standalone/test -q`
Expected: PASS — 133 passed

Ayrıca gaze regresyonu:

Run: `./.venv/bin/python -m pytest ros2_ws/src/astro_audio/test ros2_ws/src/astro_base/test standalone/test -q`
Expected: yalnızca `test_production_hardening.py::test_ready_local_offline_tts_eliminates_15s_grace_delay` düşer (önceden var olan hata, 2026-09-04'te `git stash` ile doğrulandı)

- [ ] **Step 5: Gerçek uçtan uca tur (bir kez)**

Kullanıcı gerçek API çağrısına izin verdi. Laptop mikrofonu ve hoparlörüyle:

```bash
./.venv/bin/python standalone/track.py --no-window --seconds 40 --log-interval 2
```

Mikrofona "hey astro, nasılsın" deyin. Beklenen:
- başlangıçta `🗣️  Sesli yanıt açık — uyandırma sözcüğü: 'hey astro'`
- birkaç saniye içinde hoparlörden Türkçe cevap
- durum satırlarında `fps` ~30 kalır (tur gaze'i bloklamadı)

Sonucu — gecikme, gerçekten çalıp çalmadığı, fps düşüşü olup olmadığı — not edin. Çalışmazsa `LAST_SETUP_ERROR` satırını ve terminal çıktısını kaydedin; tur bir sonraki oturumda bu çıktıdan sürdürülür.

- [ ] **Step 6: README'yi güncelle ve commit**

`standalone/README.md` içindeki çalıştırma bloğuna ekle:

```
./.venv/bin/python standalone/track.py --no-voice          # yalnizca takip
```

ve "Donanım zorunlu değil" bölümünün sonuna:

```
Sesli yanıt `OPENAI_API_KEY` ister. Anahtar yoksa program tek satır uyarı basar
ve yalnızca takip yapar — konuşma, görmenin önkoşulu değil.
```

```bash
git add standalone/voice.py standalone/track.py standalone/llm.py standalone/README.md standalone/test/test_voice.py
git commit -m "feat(voice): sesli yanit dongusunu track.py'ye bagla

Tur arka plan thread'inde kosuyor: kafa 30 Hz'te akiyor ve bir LLM turu
saniyeler suruyor, ayni thread'de kosarsa takip donar. Anahtar yoksa
program tek satir uyari basip yalnizca takip yapiyor.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Plan Sonrası

C1 bittiğinde çalışan şey: uyandırma sözcüğüyle açılan, sesli cevap veren, kafası bu sırada takibe devam eden, kendi sesine dönmeyen ve araya girilince susan bir döngü.

Sırada C2 (persona, hafıza, oturum özeti, streaming) var; `LlmClient.system_prompt` ve `VoiceLoop.session` şimdiden onun bağlanacağı yerler.

---

## Yürütme sonrası — açık kalanlar

C1 uygulandı (7 görev, 164 standalone testi yeşil). Aşağıdakiler bilinçli olarak
açık bırakıldı; sıradaki oturumun başlangıç listesi budur.

### Doğrulanmamış (ortam engeli)

- **Gerçek sağlayıcılarla hiçbir uçtan uca tur koşulmadı.** OpenAI hesabı
  `429 credit_balance_exhausted` döndü. Gerçek STT doğruluğu/gecikmesi, gerçek TTS
  çalması, gerçek turda fps etkisi ve gerçek donanımda `note_playback` zamanlaması
  hiç ölçülmedi. Kredi geldiğinde ilk iş bu.
- **ReSpeaker takılı değil.** Kanal sırası açık: ROS `multi_ch[:4]` (kanal 0-3),
  standalone `(1,2,3,4)`. İkisi birden doğru olamaz; `standalone/audio_check.py`
  ölçecek.
- **Sessizlik eşiği uzlaştırılmadı.** ROS yığını 0.38 s kullanıyor
  (`speech_recognition_node`), `UtteranceTracker` 0.8 s. 0.38 sahada çalışan bir
  değer, 0.8 akıl yürütme. Gerçek kayıtla karara bağlanmalı.
- **`SpeechDetector` eşikleri yalnızca sentetik sesle doğrulandı.** Gerçek konuşma,
  gerçek trafik ve gerçek uğultu kaydıyla yeniden ayarlanmalı.

### Bilinçli kapatılanlar

- **Barge-in varsayılan olarak kapalı** (`barge_in_enabled=False`). Boru hattında
  robotun kendi sesini insandan ayıracak hiçbir kanıt yok — yankı iptali yok,
  referans çıkarma yok, seviye karşılaştırması yok — ve TTS de konuşmadır. Açık
  bırakıldığında robot her cevabın ~0.3 sn'sinde kendini kesip kendine cevap
  veriyordu. Açılabilmesi için yankı iptali ya da çalma zarfına göre seviye
  ayrımı gerekiyor.

### Kalan küçük borçlar

- `LlmClient` testleri `model`/`temperature`/`timeout` iletimini doğrulamıyor.
- `UtteranceTracker.max_s` kapağı sessizlik dalından hiçbir testle sürülmüyor.
- `UtteranceTracker.active` ve `VoiceLoop.is_speaking` kullanılmıyor ve test
  edilmiyor — silinmeli ya da test edilmeli.
- Yalnız uyandırma sözcüğünden ibaret sözce ("hey astro") ham metni LLM'e prompt
  olarak gönderiyor; savunulabilir ama test edilmemiş.
- `build_default_loop`, sonraki bir kurucu hata verirse `AudioOutputManager`'ın
  worker thread'ini durdurmuyor (yetim daemon thread).
- `_speak`, TTS ses üretemediğinde de `session.record_robot_speech()` çağırıyor.
- Uzun bir tur (STT+LLM+TTS > 16 s) henüz çalma başlamamışsa oturumu tur ortasında
  düşürebiliyor: cevap yine çalınır ama takip sözcesi düşer.
- `pump()`, `thread.start()` hata verirse `_turn_running`'i True'da bırakır ve ses
  döngüsü süreç ömrü boyunca ölür.

### Sıradaki aşamalar

C2 (persona, hafıza, oturum özeti, streaming) → C3 (araçlar, ofis) → C4 (Realtime
S2S). `LlmClient.system_prompt` ve `VoiceLoop.session` C2'nin bağlanacağı yerler.
