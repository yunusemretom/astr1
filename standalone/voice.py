#!/usr/bin/env python3
"""Sesli yanıt döngüsü — ROS'suz.

ROS'lu tarafta bu iş beş düğüme dağılmış ve aralarındaki her şey topic:
audio_stream_node → speech_recognition_node → ai_brain_node → tts_node, artı
yankı bastırma için geri dönen `/tts/speaking`. Buradaki dosya o topolojinin
yerine geçen kablolama; karar veren nesnelerin hepsi paylaşılan kütüphanelerden
geliyor (`STTRouter`, `TTSRouter`, `AudioOutputManager`, `ConversationSession`).
"""

import io
import os
import threading
import time
import wave
from math import gcd
from typing import Any, Optional

import numpy as np

import core_path  # noqa: F401
from astro_ai.conversation_session import ConversationSession  # noqa: E402
from llm import LlmClient  # noqa: E402

# Sözce sonu eşiği. Cümle içi duraklamayı sözce sonundan ayırmalı: hece arası
# ≤0.25 s, virgül duraklaması ~0.5 s, cümle sonu daha uzun. 0.8 s ikisinin
# arasında ve tur gecikmesini hissedilir kılmıyor. Gerçek kayıtla yeniden
# ayarlanacak, o yüzden dışarıdan verilebiliyor.
DEFAULT_SILENCE_S = 0.8

# Susmayan bir kaynak (açık televizyon, uğultu) tamponu sonsuza büyütmesin.
DEFAULT_MAX_UTTERANCE_S = 10.0

# STT'nin beklediği hız. Tampon yakalama hızında tutulur (dizi modunda 16 kHz,
# stereo modda aygıtın kendi hızı — laptopta 44.1 kHz) ve dönüşüm tek noktada,
# STT'ye verilmeden hemen önce yapılır. Hızın sessizce varsayılması bu depoda
# daha önce bedel ödetti: int16 eşiği float32 akışa uygulanmıştı ve kapı her
# bloğu eliyordu.
STT_SAMPLE_RATE = 16000

WAKE_WORD = "hey astro"

# Hoparlör sustuktan sonra mikrofonun kendi sesini duymayı bırakması zaman alır.
# .env'deki ECHO_MUTE_COOLDOWN_S ile aynı sayı; bayrak anında düşerse yolda olan
# kendi ses transkribe edilir ve robot kendine cevap verir.
DEFAULT_ECHO_COOLDOWN_S = 0.65

# Robot konuşurken araya girmeyi ciddiye almadan önce görülmesi gereken kesintisiz
# konuşma. Tek blok (64 ms) hoparlör sızıntısıyla da olabilir; 0.3 s bir hecedir.
DEFAULT_BARGE_IN_S = 0.3

# Barge-in VARSAYILAN OLARAK KAPALI (bkz. review C2). `_check_barge_in`'e giden
# `is_speech`, `pump`'ın `audio.latest_speech()`'ten okuduğu VERDİNİN TA
# KENDİSİ — ve o verdi robotun kendi hoparlör çıkışını da içeren bir mikrofon
# penceresinden çıkıyor. `SpeechDetector` TTS'i insan konuşmasından ayıramaz:
# TTS zaten konuşmanın kendisi. Bu boru hattında yankı iptali yok, referans
# çıkarma yok, çalınanla mikrofonun seviye karşılaştırması yok — robotun kendi
# sesini insan sesinden ayıracak hiçbir kanıt yok. Sonuç: barge-in her cevabın
# ~0.3 sn'sinde kendi sesine tetikleniyor; `AudioOutputManager.interrupt()`
# sounddevice'ı `stream.write` ortasında gerçekten durdurmadığı için cümle
# çalmaya devam ediyor; 0.65 sn sonra soğuma bitiyor ve hâlâ çalan cevap STT'ye
# gidip yeni bir tur açıyor — robot kendiyle konuşuyor, her turda API kredisi
# harcıyor. Yarı çalışan barge-in, hiç barge-in olmamasından kötü: en azından
# yankı bastırma (echo_cooldown) tek başına robotun kendini duymasını zaten
# durduruyor. Açmak için gereken: gerçek yankı iptali (AEC) ya da çalınan sesin
# bilinen zarfına karşı bir seviye karşılaştırması — ikisi de bugün yok.
DEFAULT_BARGE_IN_ENABLED = False


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
        elif self._chunks:
            # Sözce açıkken gelen sessizlik saklanır — cümle içi duraklama sözcenin
            # parçasıdır ve atılırsa kelimeler birbirine yapışır.
            self._chunks.append(samples)
            self._samples += len(samples)

            if self._silence_started_at is None:
                self._silence_started_at = timestamp
            elif (timestamp - self._silence_started_at) >= self.silence_s:
                return self._close()
        else:
            # Sözce başlamadı: sessizlik biriktirilmez.
            return None

        # Azami süre sınırı, konuşma veya sessizlik bloğu eklendikten sonra uygulanır.
        if self._samples >= int(self.max_s * self.sample_rate):
            return self._close()
        return None

    def _close(self) -> np.ndarray:
        utterance = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        self._chunks = []
        self._samples = 0
        self._silence_started_at = None
        return utterance

    def reset(self) -> None:
        """Birikmiş sözceyi sessizce atar (sonucu döndürmez).

        Robotun kendi turu başlarken çağrılır: o ana kadar yarım kalan sözce,
        robot konuşmaya başladığı andan itibaren artık kişinin söylediği cümle
        değil — arada geçen boşluk bloklar hiç beslenmediği için görünmez
        kalıyor, o yüzden sızdırmak yerine atılması doğru olan.
        """
        self._chunks = []
        self._samples = 0
        self._silence_started_at = None


def resample_to(mono: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Polyphase (anti-alias süzgeçli) yeniden örnekler.

    Önceden çıplak `np.interp` ile ondalıklıyordu. STT modellerinin kendi ön
    işlemesinde bant sınırladığı doğru ama alakasız: 44.1 kHz -> 16 kHz'te yeni
    Nyquist 8 kHz'in üzerindeki içerik (sislemeler, fan uğultusu) STT hiç
    görmeden ÖNCE konuşma bandına katlanıyordu (aliasing), ve bir kez yanlış
    banda katlanan enerji sonradan ayrılamıyor — STT'nin kendi filtresi bunu
    telafi etmiyor. `scipy.signal.resample_poly` oranı `from_rate`/`to_rate`'in
    ortak böleniyle sadeleştirip alçak geçirgen süzülmüş bir polifaz yeniden
    örnekleme uyguluyor, bu yüzden bant dışı içerik gerçekten atılıyor.

    `scipy` burada pip bağımlılığı değil (bkz. review R2) -- bu makinede
    yalnızca `--system-site-packages` üzerinden geliyor, temiz bir kurulumda
    yok olabilir. `astro_audio/speaker_db.py`'deki `to_16k_mono` aynı ödünle
    kurulmuştu: import içeride ve tembel, yoksa `np.interp`'e düşülüyor.
    """
    if from_rate == to_rate or len(mono) == 0:
        return np.asarray(mono, dtype=np.float32)
    mono = np.asarray(mono, dtype=np.float32)
    try:
        from scipy.signal import resample_poly
    except ImportError:
        # scipy yoksa çıplak np.interp'e düşülüyor: bu katlanma (aliasing)
        # yaratır (yeni Nyquist üzerindeki içerik konuşma bandına bulaşır) --
        # ama STT yine de az bozulmuş sesle çalışır, oysa burada çökmek
        # track.py'nin ses gerektirmeyen görüntü-yalnız yolunu da beraberinde
        # götürürdü (bkz. README: "konuşma, görmenin önkoşulu değil").
        indices = np.linspace(0, mono.size - 1,
                               int(mono.size * to_rate / from_rate))
        return np.interp(indices, np.arange(mono.size), mono).astype(np.float32)
    divisor = gcd(int(from_rate), int(to_rate))
    up, down = int(to_rate) // divisor, int(from_rate) // divisor
    return resample_poly(mono, up, down).astype(np.float32)


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
        wake_word: str = WAKE_WORD,
        silence_s: float = DEFAULT_SILENCE_S,
        echo_cooldown_s: float = DEFAULT_ECHO_COOLDOWN_S,
        barge_in_s: float = DEFAULT_BARGE_IN_S,
        barge_in_enabled: bool = DEFAULT_BARGE_IN_ENABLED,
    ):
        self.audio = audio
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.output = output
        self.session = session or ConversationSession()
        self.wake_word = wake_word
        self.silence_s = float(silence_s)
        self.echo_cooldown_s = float(echo_cooldown_s)
        self.barge_in_s = float(barge_in_s)
        self.barge_in_enabled = bool(barge_in_enabled)
        self._playing = False
        self._playback_ended_at: Optional[float] = None
        self._barge_in_since: Optional[float] = None
        # Bir çalma turu için barge-in tek seferlik: kilitsiz haliyle her
        # barge_in_s'de (0.3 sn) bir yeniden tetikleniyordu çünkü barge_in_s <
        # echo_cooldown_s (0.65 sn) — soğuma hiç bitmiyor ve tetikleyen konuşma
        # hiç transkribe edilmiyordu. `note_playback(True, ...)` yeni turda açar.
        self._barge_in_latched = False
        # Sözce takipçisi **yakalama** hızıyla kurulur, STT hızıyla değil. Bloklar
        # tampondan yakalama hızında geliyor (dizi modunda 16 kHz, stereo modda
        # aygıtın kendi hızı — laptopta 44.1 kHz); STT hızını varsaymak sözce üst
        # sınırını 2.75 kat yanlış hesaplardı. Dönüşüm yalnızca STT'ye girerken.
        capture_rate = int(getattr(audio, "sample_rate", 0) or STT_SAMPLE_RATE)
        self._utterance = UtteranceTracker(sample_rate=capture_rate,
                                           silence_s=self.silence_s)
        # `LlmClient.last_error` sessizce hicbir yerde okunmuyordu (gercek bir
        # programlama hatasi robotta yalnizca "hic cevap vermiyor" olarak
        # goruluyordu), ve TTS'in basarisiz sentezi de ayni sekilde yutuluyordu.
        # Ikisi de ayni disiplinle raporlanir: kategori basina son basilan
        # mesaj tutulur, ayni sebep tekrar basilmaz -- tekrarlayan bir
        # saglayici kesintisi kafa takibi teshislerinin aktigi durum log'unu
        # bogmamali. Farkli bir sebep (ya da farkli bir kategori) her zaman basilir.
        self._last_reported: dict = {}
        self._turn_running = False
        self._thread = None
        # Halka tampon imleci: her örnek tam bir kez teslim edilsin. İki yazarı
        # var -- `pump` (ana thread) ve `_discard_turn_backlog` (tur thread'i) --
        # o yüzden kendi kilidini taşır (bkz. review R1). `AudioSource._lock`
        # kasıtlı olarak kullanılmıyor: o ses kaynağının kilidi, bunun değil.
        self._cursor = 0
        self._cursor_lock = threading.Lock()

    @property
    def is_speaking(self) -> bool:
        """Anlık durum. Zamana bağlı karar için `is_speaking_at` kullanılır."""
        if self.output is not None and getattr(self.output, "is_playing", False):
            return True
        return self._playing

    def note_playback(self, is_playing: bool, timestamp: float) -> None:
        """`AudioOutputManager`'ın çalma durumu değişince çağrılır."""
        if is_playing:
            # Yeni tur: önceki turun barge-in kilidi artık geçmişte kalır, ve o
            # ana kadar yarım kalan sözce robotun konuşmaya başlamasıyla farklı
            # bir cümleye dönüşür — sızdırmak yerine atılır.
            self._barge_in_latched = False
            self._utterance.reset()
        elif self._playing:
            self._playback_ended_at = float(timestamp)
        self._playing = bool(is_playing)

    def is_speaking_at(self, timestamp: float) -> bool:
        """Robot şu anda konuşuyor mu — yankı soğuması dahil.

        `output.is_playing` da ayrıca sorulur: yalnızca iç aynaya (`_playing`)
        bakılsaydı, `note_playback` çağrılmadan başlayan bir çalma sessizce
        bastırmasız kalırdı. Kaçırılan bildirim aşırı bastırmaya düşmeli, hiç
        bastırmamaya değil.
        """
        if self._playing:
            return True
        if self.output is not None and getattr(self.output, "is_playing", False):
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
        if not self.barge_in_enabled:
            # Varsayılan olarak kapalı — bkz. DEFAULT_BARGE_IN_ENABLED'in
            # yanındaki not. Robotun kendi sesini insan sesinden ayıracak hiçbir
            # kanıt yokken tetiklemek, robotun kendiyle konuşmasına yol açıyordu.
            return
        if self._barge_in_latched:
            # Bu tur için zaten tetiklendi. Kilitlemeden her barge_in_s'de bir
            # yeniden tetiklenip _playback_ended_at'i tazeler, ve barge_in_s
            # echo_cooldown_s'den kısa olduğundan soğuma hiç bitmezdi.
            return
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
            self._barge_in_latched = True

    def pump(self, timestamp: float) -> None:
        """Ana döngüden her karede çağrılır; iş varsa arka plana atar.

        Kafa 30 Hz'te akıyor ve bir LLM turu saniyeler sürüyor — tur bu thread'de
        koşarsa takip donar. O yüzden burada yalnızca sözce toplanır; tur kapanınca
        işi bir arka plan thread'i alır.

        Oturum ömrü de buradan sürülür (bkz. review I1): `activate_session` bir
        uyandırma sözcüğüyle çağrılıyordu ama `check_and_update_session_lifecycle`
        hiçbir yerden çağrılmıyordu — `ConversationSession._is_active` yalnızca o
        metodun içinden temizleniyor, o yüzden tek bir "hey astro" süreç ömrü
        boyunca oturumu açık bırakıyordu. `is_speaking_at` burada geçiriliyor ki
        uzun bir cevap ortasında oturum sessizlik sanılıp süresi dolmasın.
        """
        self.session.check_and_update_session_lifecycle(
            is_robot_speaking=self.is_speaking_at(timestamp))

        if self.audio is None or self._turn_running:
            return

        block = self._read_since_cursor()
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
            # Sıra kasıtlı: `_turn_running` ATILMADAN ÖNCE değil, birikinti
            # atıldıktan SONRA düşürülüyor (bkz. review R1). Tersi -- eski
            # kod -- `_turn_running`'i önce False yapıyordu; `pump` 30 Hz ana
            # thread'de çalışıyor ve bayrağı False görür görmez kendi
            # `read_since` çağrısını yapabiliyordu. `_discard_turn_backlog`
            # hemen `AudioSource._lock`'a takılıyor (yakalama callback'i
            # periyodik olarak tutuyor), o yüzden iki thread'in burada
            # gerçekten kuyruğa girdiği ölçüldü -- pencere teorik değil.
            # Araya giren `pump` birikmiş çok-saniyelik sesi TEK blok, TEK
            # verdiyle `UtteranceTracker`'a yutturuyordu -- discard'ın
            # önlemesi gereken hatanın ta kendisi.
            self._discard_turn_backlog()
            self._turn_running = False

    def _read_since_cursor(self) -> np.ndarray:
        """`self._cursor`'ı okuyup ilerletir -- tek kilit altında (bkz. R1).

        İki yazar var: `pump` (ana thread) ve `_discard_turn_backlog` (tur
        thread'i). `AudioSource._lock` burada kullanılmıyor; o ses kaynağının
        kendi kilidi, bu nesnenin imlecini korumuyor.
        """
        with self._cursor_lock:
            block, self._cursor = self.audio.read_since(self._cursor)
        return block

    def _discard_turn_backlog(self) -> None:
        """Tur boyunca biriken ses, bir sonraki sözcenin parçası değil (bkz.
        review I2).

        `_turn_running` iken imleç bilerek ilerletilmiyor (`pump` bir tur
        sürerken hiç okumuyor) — ama mikrofon dinlemeye devam ediyor, o yüzden
        turdan sonraki ilk okuma STT+LLM+TTS boyunca geçen bütün sesi TEK blok,
        TEK `is_speech` verdisiyle `UtteranceTracker`'a verirdi. Bunun iki
        sonucu vardı: `UtteranceTracker.max_s` (10.0) `UTTERANCE_BUFFER_S`
        (10.0) ile tam eşit olduğundan 10 sn'yi aşan bir tur ilk blokta sahte
        bir sözce kapatıyordu; ve tur hiç çalmadıysa (boş transkript, uyandırma
        sözcüğü yok, sağlayıcı hatası) `note_playback(True, ...)` hiç
        tetiklenmediği için `_utterance.reset()` de hiç çalışmıyor, birikinti
        bir sonraki cümleye ekleniyordu.

        İkisi de burada kapatılıyor: imleç şimdiye ilerletilip birikinti
        atılıyor, ve `_utterance` koşulsuz sıfırlanıyor (yalnızca `note_playback`
        çalma başlarken yaptığı gibi değil — çalma hiç olmasa da).
        """
        if self.audio is not None:
            self._read_since_cursor()
        self._utterance.reset()

    def stop(self) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

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
        # else: yapacak bir şey yok — is_wake_word() uyandırma sözcüğü bulamasa
        # bile normalize edilmiş metni döndürür, devam turu zaten `clean`'de
        # ihtiyacı olanı buluyor.

        self.session.record_user_speech()
        prompt = clean.strip() or text
        answer = self.llm.reply(prompt) if self.llm is not None else None
        if not answer:
            error = getattr(self.llm, "last_error", None) if self.llm is not None else None
            if error:
                self._report_once("llm", str(error))
            return None

        self._speak(answer)
        return answer

    def _report_once(self, category: str, message: str) -> None:
        """Bir kategoride ayni sebebi ikinci kez basmaz.

        `category` sebeplerin birbirini ezmesini engeller (llm ve tts ayri
        ayri kendi son mesajini hatirlar); "farkli bir sebep" her zaman
        basilir, "ayni sebep" bir daha basilmaz -- tekrarlayan bir saglayici
        kesintisi kafa takibi teshislerinin aktigi durum log'unu bogmamali.
        """
        if self._last_reported.get(category) == message:
            return
        self._last_reported[category] = message
        self._report(message)

    def _report(self, message: str) -> None:
        print(f"🗣️  {message}")

    def _speak(self, text: str) -> None:
        if self.tts is None:
            return
        generation_id = self.output.new_generation() if self.output is not None else 0
        result = self.tts.synthesize_and_play(text, generation_id=generation_id,
                                              output_manager=self.output, language="tr")
        # `synthesize_and_play` hatayi istisna olarak degil `TTSRouteResult.pcm`
        # bos/`None` olarak bildirir (kota, ag, gecersiz anahtar TTS
        # katmaninda). Kontrol edilmezse robot "konustu" sayilir ama hoparlorden
        # hicbir sey cikmaz -- "eksik parca gorunur kalir" ilkesi TTS icin de
        # gecerli, yalnizca LLM icin degil.
        if result is not None and not getattr(result, "pcm", None):
            reason = getattr(result, "fallback_reason", None) or "bilinmeyen sebep"
            self._report_once("tts", f"TTS ses üretemedi: {reason}")
        self.session.record_robot_speech()


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
        LAST_SETUP_ERROR = "OPENAI_API_KEY anahtarı yok — sesli yanıt kapalı, takip çalışmaya devam ediyor"
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

    # Bileşenlerin kendi `logger`'ı verilmezse `self._log = lambda *a: None`'a
    # düşer: "ne sounddevice ne aplay var" gibi kurulum hataları sessizce
    # yutulur ve `track.py` "Sesli yanıt açık" basıp sonra hiç konuşmaz. Bu üç
    # geri çağrı, döngü henüz kurulmadıysa (kurulum sırasında) doğrudan basar,
    # kurulduktan sonra (çalışma zamanı hataları) `loop._report_once` üzerinden
    # kategori başına tekrar etmeyecek şekilde basar. Mesaj metni bileşenin
    # kendi etiketini taşıdığı için ("[AudioOutputManager]", "[STTRouter]",
    # "[TTS ROUTER]"/"[OpenAI-TTS ...]") hangi katmanın sustuğu satırdan belli
    # olur — "anahtar yok" ve "paket eksik" satırlarından ayrı okunur.
    def _make_logger(category: str):
        def _log(level: str, message: str) -> None:
            if level not in ("error", "warn"):
                return
            loop = cell.get("loop")
            if loop is not None:
                loop._report_once(category, message)
            else:
                print(f"🗣️  {message}")
        return _log

    try:
        client = OpenAI(api_key=api_key)
        output = AudioOutputManager(on_playback_state_change=_on_playback,
                                    logger=_make_logger("cikis"))
        # `OpenAITTSEngine` bir `client` kwarg'ı almıyor — yalnızca `api_key`
        # (bkz. astro_audio/openai_tts_engine.py); brief'teki `client=client`
        # her çağrıda TypeError üretip sağlayıcı kurulumunu sessizce
        # düşürüyordu (üstteki `except Exception` bunu yutuyordu).
        edge_enabled = kwargs.pop("edge_tts_enabled", True)
        edge_engine = None
        if edge_enabled:
            try:
                from astro_audio.edge_tts_engine import EdgeTTSEngine
                edge_engine = EdgeTTSEngine(logger=_make_logger("edge_tts"))
            except Exception:
                edge_engine = None

        tts = TTSRouter(openai_tts_engine=OpenAITTSEngine(api_key=api_key),
                        edge_tts_engine=edge_engine,
                        edge_tts_enabled=bool(edge_engine is not None and edge_enabled),
                        output_manager=output,
                        logger=_make_logger("tts_router"))
        stt = STTRouter(openai_client=client, logger=_make_logger("stt"))
    except Exception as exc:
        LAST_SETUP_ERROR = f"sağlayıcılar kurulamadı: {exc}"
        return None

    LAST_SETUP_ERROR = ""
    loop = VoiceLoop(audio=audio, stt=stt, llm=LlmClient(client=client),
                     tts=tts, output=output, **kwargs)
    cell["loop"] = loop
    return loop
