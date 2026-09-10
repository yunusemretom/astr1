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
