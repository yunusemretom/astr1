#!/usr/bin/env python3
"""Sesli yanit dongusu testleri. Donanim ve ag istemez.

Sozce siniri burada test ediliyor: konusma nerede basladi, nerede bitti.
Cumle ici duraklama sozceyi kapatmamali (hece arasi <=0.25 sn, virgul
duraklamasi ~0.5 sn), sozce sonu kapatmali.
"""

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core_path  # noqa: F401,E402
from voice import UtteranceTracker, resample_to  # noqa: E402

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
        # Kapalis sessizligi sozceye dahil ettiginden, uzunlugun ayni olmasi
        # sızıntının olmadığını gösterir — _close() tamamı sıfırlamalı.
        self.assertEqual(len(second[0]), len(first[0]),
                         "ikinci sozce birincinin sesini tasiyor")

    def test_tam_ornek_sayisi_kapanista(self):
        """Sessizlik esiginin tam blok sayisini doğrulamak.

        Konusma + sessizlik duzeninde örnek sayisi hesaplanir:
        silence_s=0.8, BLOCK_S=0.064 ile kapanis koşulu:
        - 1 blok konusma (blok 0, t=0)
        - Blok 1'de sessizlik baslar (t=0.064), silence_started_at=0.064
        - Kapanis esigi: (t - 0.064) >= 0.8 → t >= 0.864
        - 0.864 / 0.064 = 13.5, yani blok 14 ilk kapanisi tetikler
        - Blok 14 (t=0.896): (0.896 - 0.064) = 0.832 >= 0.8 ✓
        - Dahil bloklar: 0-14 = 15 blok = 15360 örnek
        """
        closed = self._feed([True] * 1 + [False] * 16)
        self.assertEqual(len(closed), 1)
        self.assertEqual(len(closed[0]), 15 * BLOCK)


class ResampleAliasingTests(unittest.TestCase):
    """I4: `resample_to` 44.1kHz -> 16kHz'i çıplak `np.interp` ile ondalıklıyordu,
    anti-alias süzgeci olmadan. Docstring "STT modelleri kendi ön işlemesinde
    zaten bant sınırlıyor" diyordu ama bu alakasız -- katlanma (aliasing) burada,
    STT hiç görmeden önce oluyor ve geri alınamıyor."""

    def test_yeni_nyquistin_ustundeki_icerik_banda_katlanmiyor(self):
        """44.1 kHz'de 12 kHz'lik bir ton, 16 kHz'e (Nyquist 8 kHz) indirgenince
        tamamen bant dışında kalmalı. Duzgun suzulmus bir yeniden ornekleme
        onu neredeyse sessizlige indirir; anti-alias suzgeci olmayan dogrusal
        ara deger ise enerjiyi bandin icine katlayip tasir."""
        fs = 44100
        t = np.arange(fs) / fs  # 1 sn
        tone = np.sin(2.0 * np.pi * 12000.0 * t).astype(np.float32)  # > 8 kHz

        out = resample_to(tone, fs, 16000)

        input_rms = float(np.sqrt(np.mean(tone ** 2)))
        output_rms = float(np.sqrt(np.mean(out ** 2)))
        self.assertLess(
            output_rms, 0.1 * input_rms,
            f"bant disi icerik katlanarak tasindi (girdi rms={input_rms:.3f}, "
            f"cikti rms={output_rms:.3f}) -- anti-alias suzgeci yok")

    def test_hiz_esitse_veri_degismeden_doner(self):
        mono = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        out = resample_to(mono, 16000, 16000)

        np.testing.assert_array_equal(out, mono)

    def test_bos_girdi_bos_cikti_verir(self):
        out = resample_to(np.zeros(0, dtype=np.float32), 44100, 16000)

        self.assertEqual(len(out), 0)


class ResampleWithoutScipyTests(unittest.TestCase):
    """R2: `voice.py` modul seviyesinde `from scipy.signal import resample_poly`
    yapiyordu. `scipy` bu depoda pip bagimliligi degil -- yalnizca bu
    makinede `--system-site-packages` uzerinden geliyor (`requirements.in`da
    yoktu). Temiz bir kurulumda yoksa modul importu patlar, ve `track.py`
    `import voice`'u `--no-voice` kontrolunden ONCE calistirdigi icin
    goruntu-yalniz takip yolu da (README'nin ozellikle vaat ettigi) beraberinde
    cokerdi. Duzeltme `astro_audio/speaker_db.py::to_16k_mono`'daki desenin
    aynisi: import fonksiyon icinde ve tembel, `ImportError`'da `np.interp`'e
    dusuluyor. Bu test scipy'yi GERCEKTEN kaldirmiyor -- sys.modules'ta
    gorunmez kilip ayni etkiyi taklit ediyor."""

    def test_scipy_yokken_voice_ice_aktarilir_ve_resample_dogru_calisir(self):
        import importlib
        import sys as _sys
        from unittest import mock

        saved = {}
        for name in list(_sys.modules):
            if name == "voice" or name == "scipy" or name.startswith("scipy."):
                saved[name] = _sys.modules.pop(name)

        try:
            with mock.patch.dict(_sys.modules, {"scipy": None, "scipy.signal": None}):
                voice_mod = importlib.import_module("voice")  # patlamamali

                mono = np.full(44100, 0.5, dtype=np.float32)  # sabit genlik
                out = voice_mod.resample_to(mono, 44100, 16000)

                self.assertGreater(len(out), 0,
                                   "scipy yokken resample_to bos cikti dondurdu")
                self.assertAlmostEqual(
                    float(np.mean(out)), 0.5, places=2,
                    msg="scipy yokken np.interp yedegi genligi bozdu")
        finally:
            for name in list(_sys.modules):
                if name == "voice" or name == "scipy" or name.startswith("scipy."):
                    del _sys.modules[name]
            _sys.modules.update(saved)


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


_UNSET = object()


def _voice(stt=_UNSET, llm=_UNSET, tts=_UNSET, output=_UNSET, **kwargs):
    """Sahte saglayicilarla bir VoiceLoop kurar.

    Bir sentinel kullanir, `None` degil: bazi testler bilerek `llm=None` ya
    da `tts=None` gecirip o parcanin gercekten eksik oldugu durumu sinar --
    `x or _FakeX()` bunu sessizce varsayilana cevirip testi anlamsizlastirirdi.
    """
    from voice import VoiceLoop

    if stt is _UNSET:
        stt = _FakeStt()
    if llm is _UNSET:
        llm = _FakeLlm()
    if tts is _UNSET:
        tts = _FakeTts()
    if output is _UNSET:
        output = _FakeOutput()
    return VoiceLoop(audio=None, stt=stt, llm=llm, tts=tts, output=output, **kwargs)


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

    def test_devam_turunde_de_turkce_normallestirme_uygulanir(self):
        """is_wake_word() uyandirma sozcugu yokken de normallestirilmis metni
        dondurur (orn. 'ahlatta' -> "Ahlat'ta"); devam turu bunu ham STT
        ciktisiyla ezmemeli, yoksa fonetik duzeltme yalnizca oturumu acan
        soylemde calisir."""
        llm = _FakeLlm()
        loop = _voice(stt=_FakeStt("hey astro merhaba"), llm=llm)
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        loop.stt = _FakeStt("ahlatta oturuyorum")
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertEqual(llm.prompts[1], "Ahlat'ta oturuyorum",
                         f"devam turu normallestirilmemis ham metinle gitti: {llm.prompts[1]!r}")

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


class EchoAndBargeInTests(unittest.TestCase):
    """Robot kendini duymamali, ama kullanici araya girince susmali."""

    def _loop(self, output=None, barge_in_enabled=False):
        return _voice(output=output or _FakeOutput(), echo_cooldown_s=0.65,
                      barge_in_enabled=barge_in_enabled)

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
        """Barge-in varsayılan olarak kapalı (bkz. C2); burada bilerek açılıyor
        çünkü test edilen şey tetikleme mantığının kendisi, varsayılan değil."""
        output = _FakeOutput()
        loop = self._loop(output=output, barge_in_enabled=True)
        loop.note_playback(True, timestamp=10.0)

        t = 10.0
        for _ in range(6):                       # 0.38 sn kesintisiz konusma
            loop.on_block(True, _block(), timestamp=t)
            t += BLOCK_S

        self.assertGreaterEqual(output.interrupts, 1, "barge-in calismadi")

    def test_robot_konusurken_gurultu_calmayi_kesmez(self):
        output = _FakeOutput()
        loop = self._loop(output=output, barge_in_enabled=True)
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

    def test_barge_in_bir_kez_tetiklenir_sonrasinda_konusma_transkribe_edilir(self):
        """Bulgu 1: kilitsiz barge-in her 0.3 sn'de bir kendini yeniden tetikler
        (barge_in_s < echo_cooldown_s), soguma hic bitmez ve tetikleyen konusma
        hicbir zaman transkribe edilmez. Kilitliyken tetik bir kez olmali ve
        soguma bitince ayni konusma sozceye akmali. Barge-in varsayilan olarak
        kapali (bkz. C2); kilit mantigini sinamak icin burada aciliyor."""
        output = _FakeOutput()
        loop = self._loop(output=output, barge_in_enabled=True)
        loop.note_playback(True, timestamp=10.0)

        closed = []
        t = 10.0
        for _ in range(40):                      # ~2.56 sn kesintisiz konusma
            out = loop.on_block(True, _block(), timestamp=t)
            if out is not None:
                closed.append(out)
            t += BLOCK_S
        for _ in range(20):                      # sessizlik: sozce kapanmali
            out = loop.on_block(False, _block(), timestamp=t)
            if out is not None:
                closed.append(out)
            t += BLOCK_S

        self.assertEqual(output.interrupts, 1, "barge-in defalarca tetiklendi (kilit yok)")
        self.assertEqual(len(closed), 1, "tetikleyen konusma hic transkribe edilmedi")

    def test_varsayilan_ayarlarla_barge_in_hic_tetiklenmez(self):
        """C2: bu boru hattinda robotun kendi sesini insan sesinden ayiracak
        hicbir kanit yok (yanki iptali yok, referans cikarma yok, seviye
        karsilastirmasi yok) -- SpeechDetector TTS'i konusma sayar, ve yarim
        calisan barge-in robotun kendiyle konusmasina yol aciyordu (cevabin
        ~0.3 sn'sinde kendi sesine tetiklenip, interrupt() sesi gercekten
        kesmeyince ayni cevap 0.65 sn sonra STT'ye gidiyordu). Bu yuzden
        varsayilan kapali: burada `barge_in_enabled` hic verilmiyor."""
        output = _FakeOutput()
        loop = self._loop(output=output)
        loop.note_playback(True, timestamp=10.0)

        t = 10.0
        for _ in range(40):                      # ~2.56 sn kesintisiz "konusma"
            loop.on_block(True, _block(), timestamp=t)
            t += BLOCK_S

        self.assertEqual(output.interrupts, 0,
                         "barge-in varsayilan ayarlarla yine de calmayi kesti")

    def test_output_playing_bayragi_note_playback_olmadan_da_yankiyi_bastirir(self):
        """Bulgu 2: `note_playback` cagrilmazsa `is_speaking_at` yalnizca ic
        aynaya bakiyordu ve yanki bastirma sessizce kapaniyordu. Kacirilan
        bildirim asiri bastirmaya dusmeli, hic bastirmamaya degil."""
        output = _FakeOutput()
        output.is_playing = True
        loop = self._loop(output=output)

        self.assertTrue(loop.is_speaking_at(123.0), "output.is_playing yoksayildi")

    def test_robot_konusmaya_baslayinca_yarim_sozce_atilir(self):
        """Bulgu 3: robotun turu baslamadan once yarim kalmis sozce, tur bitince
        yeni cumleyle birlesmemeli. Aradaki 5 saniyelik bosluk UtteranceTracker'a
        gorunmuyor (blok akisi kesilmiyor, sadece beslenmiyor), o yuzden robot
        konusmaya baslarken tampon atilmali."""
        loop = self._loop()

        t = 0.0
        loop.on_block(True, _block(value=0.9), timestamp=t)
        t += BLOCK_S
        loop.on_block(True, _block(value=0.9), timestamp=t)
        t += BLOCK_S

        loop.note_playback(True, timestamp=10.0)
        loop.note_playback(False, timestamp=15.0)          # 5 sn sonra

        closed = []
        t = 15.0 + loop.echo_cooldown_s + 0.01
        for is_speech in [True] * 10 + [False] * 16:
            out = loop.on_block(is_speech, _block(value=0.2), timestamp=t)
            if out is not None:
                closed.append(out)
            t += BLOCK_S

        self.assertEqual(len(closed), 1)
        self.assertTrue(bool(np.all(closed[0] == 0.2)),
                        "robot konusmadan onceki blok yeni sozceye sizdi")


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

        Kaynagi metin olarak taramak yerine (eskiden `inspect.getsource` +
        alt dize kontrolu -- adini degistirip cagirmayi unutan bir refactor'u
        yakalamazdi) sahte bir cikis yoneticisiyle gercekten kurup geri
        cagriyi tetikliyor ve dongunun durumunun degistigini dogruluyor."""
        import sys
        import types
        from unittest import mock

        import voice

        captured: dict = {}

        class _StubOutput:
            def __init__(self, on_playback_state_change=None, logger=None):
                captured["on_playback_state_change"] = on_playback_state_change
                self.is_playing = False

        class _StubProvider:
            def __init__(self, **kwargs):
                pass

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = lambda api_key=None: object()
        fake_output_mod = types.ModuleType("astro_audio.audio_output_manager")
        fake_output_mod.AudioOutputManager = _StubOutput
        fake_tts_engine_mod = types.ModuleType("astro_audio.openai_tts_engine")
        fake_tts_engine_mod.OpenAITTSEngine = _StubProvider
        fake_stt_router_mod = types.ModuleType("astro_audio.stt_router")
        fake_stt_router_mod.STTRouter = _StubProvider
        fake_tts_router_mod = types.ModuleType("astro_audio.tts_router")
        fake_tts_router_mod.TTSRouter = _StubProvider

        with mock.patch.dict(sys.modules, {
            "openai": fake_openai,
            "astro_audio.audio_output_manager": fake_output_mod,
            "astro_audio.openai_tts_engine": fake_tts_engine_mod,
            "astro_audio.stt_router": fake_stt_router_mod,
            "astro_audio.tts_router": fake_tts_router_mod,
        }):
            loop = voice.build_default_loop(audio=None, api_key="fake-key")

        self.assertIsNotNone(loop, voice.LAST_SETUP_ERROR)
        self.assertIn("on_playback_state_change", captured,
                      "AudioOutputManager calma durumu geri cagrisiyla kurulmuyor")

        self.assertFalse(loop.is_speaking_at(0.0))
        captured["on_playback_state_change"](True)

        self.assertTrue(
            loop.is_speaking_at(0.0),
            "AudioOutputManager'in calma geri cagrisi VoiceLoop.note_playback'e baglanmadi")

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


class _FakeLlmWithError:
    """LLM sahtesi: cevap uretemez ve sebebini last_error'a yazar."""

    def __init__(self, error="beklenmeyen cevap bicimi: boom"):
        self.prompts = []
        self.last_error = None
        self._error = error

    def reply(self, user_text):
        self.prompts.append(user_text)
        self.last_error = self._error
        return None


class LlmErrorReportingTests(unittest.TestCase):
    """`LlmClient.last_error` hic okunmuyordu: gercek bir programlama hatasi
    (yanlis yazilmis bir kwarg gibi) yutuluyor ve robotta yalnizca "hic cevap
    vermiyor, nedeni belirsiz" olarak goruluyordu. VoiceLoop artik bu sebebi
    bir kez basiyor -- her turda degil, cunku tekrarlayan bir saglayici kesintisi
    durum log'unu bogmamali."""

    def test_llm_hata_verince_sebep_bir_kez_basilir(self):
        llm = _FakeLlmWithError("kota asildi")
        printed = []
        loop = _voice(stt=_FakeStt("hey astro selam"), llm=llm)
        loop._report = lambda msg: printed.append(msg)

        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)
        loop.stt = _FakeStt("hey astro tekrar")
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertEqual(printed.count("kota asildi"), 1,
                         "ayni sebep birden fazla kez basildi")

    def test_farkli_hata_yeniden_basilir(self):
        llm = _FakeLlmWithError("kota asildi")
        printed = []
        loop = _voice(stt=_FakeStt("hey astro selam"), llm=llm)
        loop._report = lambda msg: printed.append(msg)

        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)
        llm._error = "ag hatasi"
        loop.stt = _FakeStt("hey astro tekrar")
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertEqual(printed, ["kota asildi", "ag hatasi"],
                         "farkli sebep basilmadi")

    def test_last_error_yoksa_hic_basilmaz(self):
        llm = _FakeLlm(answer=None)
        printed = []
        loop = _voice(stt=_FakeStt("hey astro selam"), llm=llm)
        loop._report = lambda msg: printed.append(msg)

        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertEqual(printed, [])

    UTTERANCE = np.full(16000, 0.2, dtype=np.float32)


class _FakeTtsResult:
    """`TTSRouteResult`'in testte ihtiyaç duyulan tek alanı: `pcm` boş/None
    olunca `_speak` bunu bir sentez hatası sayar, `fallback_reason`'ı basar."""

    def __init__(self, pcm=None, fallback_reason="quota_exceeded"):
        self.pcm = pcm
        self.fallback_reason = fallback_reason


class _FakeTtsRouter:
    """Gercek `TTSRouter` gibi bir `TTSRouteResult` dondurur (`_FakeTts` gibi
    None degil) -- boylece `_speak`'in sonucu inceleyip incelemedigi test
    edilebilir."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def synthesize_and_play(self, text, generation_id, output_manager=None,
                            language="tr", realtime_fallback_reason=None):
        self.calls.append(text)
        return self._result


class TtsErrorReportingTests(unittest.TestCase):
    """`_speak` `TTSRouteResult`'i hic incelemiyordu: TTS kota/ag/anahtar
    seviyesinde basarisiz olunca (istisna degil, bos `pcm`) robot "konustu"
    sayiliyor ama hoparlorden hicbir sey cikmiyordu -- eksik parca gorunmuyordu.
    LLM hatalari icin kurulan "ayni sebep bir kez" disiplini burada da gecerli."""

    UTTERANCE = np.full(16000, 0.2, dtype=np.float32)

    def test_bos_pcm_basarisizligi_bir_kez_basilir(self):
        tts = _FakeTtsRouter(_FakeTtsResult(pcm=None, fallback_reason="quota_exceeded"))
        printed = []
        loop = _voice(stt=_FakeStt("hey astro selam"), tts=tts)
        loop._report = lambda msg: printed.append(msg)

        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)
        loop.stt = _FakeStt("hey astro tekrar")
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        quota_prints = [p for p in printed if "quota_exceeded" in p]
        self.assertEqual(len(quota_prints), 1,
                         "ayni TTS sebebi birden fazla kez basildi")

    def test_farkli_tts_sebebi_yeniden_basilir(self):
        tts = _FakeTtsRouter(_FakeTtsResult(pcm=None, fallback_reason="quota_exceeded"))
        printed = []
        loop = _voice(stt=_FakeStt("hey astro selam"), tts=tts)
        loop._report = lambda msg: printed.append(msg)

        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)
        tts._result = _FakeTtsResult(pcm=None, fallback_reason="network_unavailable")
        loop.stt = _FakeStt("hey astro tekrar")
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertEqual(len(printed), 2, f"ikinci farkli sebep basilmadi: {printed}")
        self.assertIn("quota_exceeded", printed[0])
        self.assertIn("network_unavailable", printed[1])

    def test_basarili_sentezde_hicbir_sey_basilmaz(self):
        tts = _FakeTtsRouter(_FakeTtsResult(pcm=b"\x00\x00" * 480))
        printed = []
        loop = _voice(stt=_FakeStt("hey astro selam"), tts=tts)
        loop._report = lambda msg: printed.append(msg)

        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertEqual(printed, [])

    def test_llm_ve_tts_hatalari_ayri_kategoride_tutulur(self):
        """`_report_once` kategori bazlidir: llm hatasi tts hatasini ezmemeli."""
        tts = _FakeTtsRouter(_FakeTtsResult(pcm=None, fallback_reason="quota_exceeded"))
        printed = []
        loop = _voice(stt=_FakeStt("hey astro selam"),
                     llm=_FakeLlmWithError("quota_exceeded"), tts=tts)
        loop._report = lambda msg: printed.append(msg)

        # LLM hatasi: cevap uretilemez, _speak hic cagrilmaz.
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)
        self.assertEqual(printed, ["quota_exceeded"])

        # Ayni metin ("quota_exceeded") simdi TTS tarafindan da basarisiz --
        # farkli kategoride oldugu icin yine de basilmali.
        loop.llm = _FakeLlm(answer="Iyiyim.")
        loop.stt = _FakeStt("hey astro tekrar")
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertEqual(len(printed), 2,
                         "tts kategorisi llm kategorisinin sebebiyle karistirildi")


def _make_logger_smoke_test_source():
    import voice
    import inspect
    return inspect.getsource(voice.build_default_loop)


class ProviderVisibilityTests(unittest.TestCase):
    """Bulgu 1: `AudioOutputManager`/`STTRouter`/`TTSRouter` `logger=` almadan
    kurulunca kendi `self._log`'lari no-op'a duser -- "ne sounddevice ne aplay
    var" gibi kurulum hatalari sessizce yutulur ve track.py "Sesli yanit acik"
    basip sonra hic konusmaz. `build_default_loop` gercek donanimla calistigi
    icin (AudioOutputManager gercek ALSA/sounddevice yoklamasi yapar) burada
    onu calistirmiyoruz -- her ucu de `logger=` ile kurdugunu kaynaktan
    dogruluyoruz, `on_playback_state_change`/`note_playback` baglantisini
    kontrol eden mevcut `test_calma_bildirimi_donguye_baglanir` ile ayni
    yontem."""

    def test_uc_bileşen_de_logger_ile_kuruluyor(self):
        source = _make_logger_smoke_test_source()

        self.assertIn("logger=_make_logger(\"cikis\")", source,
                      "AudioOutputManager logger olmadan kuruluyor")
        self.assertIn("logger=_make_logger(\"tts_router\")", source,
                      "TTSRouter logger olmadan kuruluyor")
        self.assertIn("logger=_make_logger(\"stt\")", source,
                      "STTRouter logger olmadan kuruluyor")

    def test_logger_hata_ve_uyariyi_bilgiyi_degil_iletir(self):
        """`_make_logger` yalnizca error/warn seviyesini iletmeli -- yoksa
        rutin bilgi satirlari da durum log'unu bogar."""
        import voice

        printed = []
        cell = {}

        def _make_logger(category):
            def _log(level, message):
                if level not in ("error", "warn"):
                    return
                loop = cell.get("loop")
                if loop is not None:
                    loop._report_once(category, message)
                else:
                    printed.append(message)
            return _log

        log = _make_logger("cikis")
        log("info", "rutin bilgi -- basilmamali")
        log("error", "ne sounddevice ne aplay var")

        self.assertEqual(printed, ["ne sounddevice ne aplay var"])

    def test_kurulum_sonrasi_calisma_zamani_hatasi_donguden_gecer(self):
        """Kurulum bittikten sonra (cell['loop'] doluyken) bir bilesenin
        error logu `loop._report_once` uzerinden, kategori basina bir kez
        basilmali."""
        loop = _voice()
        printed = []
        loop._report = lambda msg: printed.append(msg)
        cell = {"loop": loop}

        def _make_logger(category):
            def _log(level, message):
                if level not in ("error", "warn"):
                    return
                cell["loop"]._report_once(category, message)
            return _log

        log = _make_logger("stt")
        log("error", "STT gecici olarak kullanilamiyor")
        log("error", "STT gecici olarak kullanilamiyor")
        log("error", "STT anahtari gecersiz")

        self.assertEqual(printed, ["STT gecici olarak kullanilamiyor",
                                   "STT anahtari gecersiz"])


class _FakeAudioSource:
    """`pump()`'in ihtiyac duydugu kadari: sample_rate, read_since, latest_speech.

    Gercek `AudioSource` yerine gecer -- donanim yok, sadece cagrilari sayar.
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.read_calls = 0
        self._cursor_seen = []

    def read_since(self, cursor):
        self.read_calls += 1
        self._cursor_seen.append(cursor)
        return _block(), cursor + BLOCK

    def latest_speech(self, now):
        return None


class _SlowLlm:
    """LLM sahtesi: `reply()` bir `release` Event'i set edilene (ya da suresi
    dolana) kadar bloke olur -- bir LLM turunun saniyeler surebildigini simule
    eder."""

    def __init__(self, answer="Iyiyim.", started=None, release=None, block_s=None):
        self.answer = answer
        self.prompts = []
        self.last_error = None
        self._started = started
        self._release = release
        self._block_s = block_s

    def reply(self, user_text):
        self.prompts.append(user_text)
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            self._release.wait(timeout=self._block_s)
        elif self._block_s is not None:
            time.sleep(self._block_s)
        return self.answer


class PumpConcurrencyTests(unittest.TestCase):
    """Bulgu 2: kafa 30 Hz'te akiyor (~33 ms cerceve butcesi), bir LLM turu
    saniyeler surebiliyor. `pump()` turu bir arka plan thread'ine atmazsa ya
    da `_turn_running` kontrolu bozulursa takip sessizce donar. Buradaki
    testler gercek donanim/ag kullanmadan bu garantiyi kilitler."""

    def _armed_loop(self):
        started = threading.Event()
        release = threading.Event()
        llm = _SlowLlm(started=started, release=release)
        loop = _voice(stt=_FakeStt("hey astro selam"), llm=llm)
        loop.audio = _FakeAudioSource()
        loop.on_block = lambda is_speech, block, timestamp: self.UTTERANCE
        return loop, started, release

    UTTERANCE = np.full(16000, 0.2, dtype=np.float32)

    def test_pump_uzun_turu_arka_plana_atar_ve_hemen_doner(self):
        loop, started, release = self._armed_loop()

        t0 = time.monotonic()
        loop.pump(0.0)
        elapsed = time.monotonic() - t0

        self.assertLess(elapsed, 0.2,
                        f"pump() turu blokladi ({elapsed:.3f}s) -- 30 Hz gaze donuk kalir")
        self.assertTrue(started.wait(timeout=1.0), "arka plan thread hic baslamadi")
        self.assertTrue(loop._turn_running, "tur hala surerken _turn_running False")

        release.set()
        loop.stop()

        self.assertFalse(loop._turn_running, "tur bitince _turn_running True kaldi")

    def test_tur_calisirken_ikinci_pump_yeni_tur_baslatmaz(self):
        loop, started, release = self._armed_loop()

        loop.pump(0.0)
        self.assertTrue(started.wait(timeout=1.0))
        first_thread = loop._thread
        reads_before = loop.audio.read_calls
        cursor_before = loop._cursor

        loop.pump(1.0)  # tur hala calisirken

        self.assertIs(loop._thread, first_thread,
                      "tur calisirken pump ikinci bir thread basladi")
        self.assertEqual(loop.audio.read_calls, reads_before,
                         "tur calisirken audio yine de okundu -- imlec ilerledi")
        self.assertEqual(loop._cursor, cursor_before,
                         "tur calisirken imlec okunmamis sesin otesine ilerledi")

        release.set()
        loop.stop()

    def test_stop_tur_bitmeden_zaman_asimiyla_doner(self):
        """`stop()` `thread.join(timeout=2.0)` kullanir. Bunu gercekten
        sinamak icin LLM'in bu suredEn UZUN bloke olmasi gerekiyor -- bu
        yuzden bu test kasitli olarak ~2 sn suruyor (gercek zaman asimini
        tetiklemenin tek yolu gercekten beklemek)."""
        release = threading.Event()  # hic set edilmeyecek
        llm = _SlowLlm(release=release, block_s=5.0)
        loop = _voice(stt=_FakeStt("hey astro selam"), llm=llm)
        loop.audio = _FakeAudioSource()
        loop.on_block = lambda is_speech, block, timestamp: self.UTTERANCE

        loop.pump(0.0)
        self.assertTrue(loop._turn_running)

        t0 = time.monotonic()
        loop.stop()
        elapsed = time.monotonic() - t0

        self.assertLess(elapsed, 3.0,
                        f"stop() 2 sn'lik join timeout'unu asip surece kadar bekledi ({elapsed:.2f}s)")


class SessionLifecycleTests(unittest.TestCase):
    """I1: `VoiceLoop` `session.activate_session()` cagiriyordu ama
    `session.check_and_update_session_lifecycle()`'i hic cagirmiyordu. ROS
    tarafi bunu bir zamanlayicidan surer (ai_brain_node, astro_realtime_node);
    burada suren yoktu, o yuzden `_is_active` hicbir zaman temizlenmiyordu.
    Tek bir "hey astro" tum surec omru boyunca odadaki her konusma-siniflandirilmis
    sesi STT+LLM+TTS'e yolluyordu -- ne uyandirma sozcugu ne zaman asimi, sinirsiz
    harcama. `session.base_timeout_s` (16 sn) tanimliydi ama hicbir zaman
    calismiyordu."""

    UTTERANCE = np.full(16000, 0.2, dtype=np.float32)

    def test_pump_oturum_omrunu_surer_ve_suresi_dolan_oturum_sozceyi_dusurur(self):
        from astro_ai.conversation_session import ConversationSession

        session = ConversationSession(base_timeout_s=0.02, gaze_extension_s=0.0)
        loop = _voice(stt=_FakeStt("hey astro selam"), session=session)

        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)
        self.assertTrue(session.is_active(), "oturum wake word ile acilmadi")

        time.sleep(0.05)
        loop.pump(0.0)  # audio yok -- yalnizca oturum omru kontrolu calismali

        self.assertFalse(
            session.is_active(),
            "pump() oturum omrunu hic surmedi -- tek bir uyandirma sozcugu "
            "surecin tamami boyunca oturumu acik birakiyordu")

        loop.stt = _FakeStt("bugun hava nasil")  # uyandirma sozcugu yok
        said = loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        self.assertIsNone(
            said, "suresi dolmus bir oturumdan sonra uyandirma sozcuksuz sozce "
            "yine de cevaplandi")

    def test_robot_konusurken_oturum_suresi_dolmaz(self):
        """Uzun bir cevap ortasinda oturum kapanmamali -- `pump` `is_robot_speaking`
        bayragini gecirmezse `check_and_update_session_lifecycle` robotun kendi
        cevabini "sessizlik" sanip zaman asimini isletir."""
        from astro_ai.conversation_session import ConversationSession

        session = ConversationSession(base_timeout_s=0.02, gaze_extension_s=0.0)
        loop = _voice(stt=_FakeStt("hey astro selam"), session=session)
        loop.handle_utterance(self.UTTERANCE, sample_rate=16000)

        output = _FakeOutput()
        output.is_playing = True
        loop.output = output

        time.sleep(0.05)
        loop.pump(0.0)

        self.assertTrue(session.is_active(),
                        "robot konusurken oturum suresi doldu")


class TurnBacklogTests(unittest.TestCase):
    """I2: `_turn_running` True iken imlec bilerek ilerletilmiyor (bkz. `pump`).
    Bir tur (STT+LLM+TTS) saniyeler surebiliyor; turdan sonraki ilk `pump` o
    araligi TEK blok, TEK `is_speech` verdisiyle `UtteranceTracker`'a veriyordu.
    Iki sonuc: `UtteranceTracker.max_s` (10.0) `UTTERANCE_BUFFER_S` (10.0) ile
    tam esit oldugundan 10 sn'yi asan bir tur ilk blokta sahte bir sozce
    kapatiyordu; ve tur hic calmadiysa (bos transkript, wake word yok,
    saglayici hatasi) `note_playback(True, ...)` hic tetiklenmedigi icin
    `_utterance.reset()` de hic calismiyor, birikinti bir sonraki cumleye
    ekleniyordu. Fix: tur biter bitmez birikinti atiliyor (imlec ilerletiliyor)
    ve `_utterance` da kosulsuz sifirlaniyor."""

    class _WriteTrackingAudio:
        """Gercek halka tampon degil: yalnizca `_written` ilerler, `read_since`
        cursor'dan sonraki HER SEYI teslim eder -- turun arkasinda biriken
        sesin buyuklugunu dogrudan gozlemlemek icin yeterli."""

        def __init__(self, sample_rate=16000):
            self.sample_rate = sample_rate
            self._written = 0

        def push(self, n_samples: int) -> None:
            self._written += n_samples

        def read_since(self, cursor: int):
            n_new = max(0, self._written - cursor)
            return np.zeros(n_new, dtype=np.float32), self._written

        def latest_speech(self, now):
            return None

    def test_tur_bitince_cursor_birikintiyi_atlar(self):
        loop = _voice(stt=_FakeStt("hey astro selam"))
        audio = self._WriteTrackingAudio()
        loop.audio = audio
        loop._cursor = 0
        audio.push(9 * 16000)  # tur surerken 9 sn ses birikti (imlec ilerlemedi)

        loop._run_turn(np.zeros(1600, dtype=np.float32), 16000)

        self.assertEqual(loop._cursor, audio._written,
                         "tur sonrasi birikinti bir sonraki pompaya akiyor")
        leftover, _ = audio.read_since(loop._cursor)
        self.assertEqual(len(leftover), 0, "birikinti hala okunabilir durumda")

    def test_calmasiz_tur_sonunda_da_sozce_takipcisi_sifirlanir(self):
        """LLM yok -> bu turda hicbir zaman cevap/calma olmuyor, dolayisiyla
        `note_playback(True, ...)` hic tetiklenmiyor."""
        loop = _voice(llm=None)
        loop.audio = _FakeAudioSource()

        # Turdan once yarim kalmis bir sozce biriktir.
        loop.on_block(True, _block(value=0.9), timestamp=0.0)
        loop.on_block(True, _block(value=0.9), timestamp=BLOCK_S)
        self.assertTrue(loop._utterance.active, "on kosul kurulmadi: sozce hic acilmadi")

        loop._run_turn(np.zeros(1600, dtype=np.float32), 16000)

        self.assertFalse(
            loop._utterance.active,
            "calmasiz tur bitince eski yarim sozce hala biriktirilmis durumda "
            "-- bir sonraki cumleyle birlesecekti")


class BacklogDiscardOrderingTests(unittest.TestCase):
    """R1: `_run_turn` `_turn_running`'i False yapip SONRA birikintiyi
    atiyordu. `pump` ana thread'de (30 Hz) kosuyor ve `_turn_running` iken
    erken donuyor -- ama iki satir arasindaki pencerede bayragi zaten False
    gorup kendi `read_since` cagrisini yapabilir, turun bastan sona biriken
    sesini TEK blok TEK verdiyle `UtteranceTracker`'a yutturur. Bu, discard'in
    var olma sebebinin ta kendisi (bkz. TurnBacklogTests / I2). Duzeltme iki
    satiri yer degistiriyor: bayrak, birikinti GERCEKTEN atildiktan sonra
    dusuyor. Gercek thread donusumunu belirlenimli yakalamak yerine
    `_discard_turn_backlog`'u gozlemlenebilir kiliyoruz: calistigi anda
    `_turn_running`'in hala True olup olmadigini kaydediyoruz."""

    def test_discard_calisirken_turn_running_hala_true(self):
        loop = _voice(stt=_FakeStt("hey astro selam"))
        loop.audio = _FakeAudioSource()

        observed_turn_running_at_discard = []
        original_discard = loop._discard_turn_backlog

        def _spy_discard():
            observed_turn_running_at_discard.append(loop._turn_running)
            original_discard()

        loop._discard_turn_backlog = _spy_discard
        loop._turn_running = True  # pump() bir tur baslattigini boyle isaretler

        loop._run_turn(np.zeros(1600, dtype=np.float32), 16000)

        self.assertEqual(
            observed_turn_running_at_discard, [True],
            "_discard_turn_backlog calisirken _turn_running zaten False'tu -- "
            "pump() araya girip henuz atilmamis birikintiyi okuyabilirdi")
        self.assertFalse(loop._turn_running,
                         "tur bitince _turn_running True kaldi")


if __name__ == "__main__":
    unittest.main()
