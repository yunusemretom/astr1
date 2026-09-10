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
