# ROS'suz sesli yanıt döngüsü — C1 tasarımı

Tarih: 2026-09-04
Durum: onaylandı, uygulamaya hazır
Kapsam: C1 (ses G/Ç + tur döngüsü iskeleti)

## Neden

`standalone/` bugün görüp duyuyor ama konuşmuyor. ROS'lu yığında sesli yanıt beş
düğüme dağılmış durumda ve aralarındaki her şey topic:

```
audio_stream_node ──/audio/speech_audio──> speech_recognition_node ──/speech/text──> ai_brain_node
       │                  /audio/vad                                                      │
       └──/audio/doa──> social_gaze_node                                            /tts/say
                                    ▲                                                     ▼
                            /robot/look_target ◄──────────────────────────────────── tts_node
                                                        /tts/speaking ──> (yankı bastırma, barge-in)
```

Taşınacak olan bu topolojinin **kendisi değil**. `ai_brain_node._on_speech`
gerçekte orkestrasyon: barge-in, konuşana bak, persona anahtarı, wake word,
oturum, LLM. Ağır işin tamamı zaten ROS'suz kütüphanelerde duruyor — ölçüldü,
`rclpy` referansı sıfır:

| modül | satır | rclpy |
|---|---|---|
| `stt_router`, `tts_router`, `tts_orchestrator`, `audio_output_manager` | ~1630 | 0 |
| `provider_registry`, `persona_engine`, `memory_manager`, `conversation_session` | ~2770 | 0 |
| `voice_recognizer`, `speaker_db` | ~640 | 0 |

Yani gaze'de olduğu gibi: **beyin paylaşılır, düğüm katmanı atılır, yerine
kablolama yazılır.** İkinci bir kopya yazılmaz — bu depo o bedeli bir kez ödedi
(firmware kopyası 535 satırın 147'sinde ayrışmış ve her açıyı 1.73 kat büyük
uyguluyordu).

Tek istisna: **LLM çağrısı `ai_brain_node` içine gömülü**, 12 ayrı çağrı
noktasında, yeniden kullanılabilir bir istemci yok. Onu çıkarmak gerekiyor.

## Kapsam

C1 bitince şu çalışır:

> "hey astro, nasılsın" → sesli yanıt. Kafa bu sırada takibe devam eder, robot
> kendi sesine dönmez, kullanıcı araya girince robot susar.

**C1 dışında** (kendi tasarımlarıyla gelecek): persona, hafıza, oturum özeti,
tekrar koruması, token streaming (C2); tool calling, hava durumu, hatırlatma,
kafa jestleri, ofis entegrasyonları (C3); Realtime S2S (C4).

Bu sınır bilinçli: C1'de **gerçekten yeni tasarım** var (tek mikrofonun
paylaşımı, bloklamayan tur, yankı/barge-in), C2–C4 büyük ölçüde mevcut ROS'suz
kütüphanelerin kablolanması. Yeni tasarımı, kablolamadan ayrı doğrulamak
istiyoruz.

## Mimari kararlar

### K1 — Ses döngüsü `track.py` ile aynı process'te, kendi thread'lerinde

Gerekçe:

- Kafa, robot konuşurken de takip etmeli. LLM çağrısı saniyeler sürüyor; 30 Hz'lik
  gaze döngüsü (kare bütçesi 33 ms) bloklanamaz.
- ALSA aynı yakalama aygıtını iki process'e genelde açtırmaz. ReSpeaker tek.
- Ayrı process + IPC, ROS'tan kaçma sebebimizin aynısını geri getirirdi.

### K2 — Tek yakalama, halka tampon, çekme (push değil)

`AudioSource` yakalamayı tek elde tutar ve ham bloklardan sabit boyutlu bir halka
tampon besler. Gaze bugünkü gibi `latest_doa_deg` / `latest_speech` ile okur; ses
döngüsü kendi thread'inde tampondan pencere çeker.

Reddedilen alternatif — **callback fan-out** (`AudioSource.subscribe(fn)`): yavaş
bir abone (STT saniyeler sürer) `sounddevice` callback thread'ini bloklar ve ses
düşer. Düzeltmek için abone başına kuyruk gerekir; o da kuyruk üstüne kuyruk
koymuş halka tampondur.

Yoklama maliyeti önemsiz: ses döngüsü zaten blok hızında (64 ms) uyanıyor.

### K3 — `SpeechDetector` üç iş birden yapar

2026-09-04'te gaze için yazılan `astro_audio.speech_detector` C1'de yeniden
kullanılır:

1. gaze için gürültü kapısı (mevcut kullanım),
2. **sözce sınırı** — konuşma başladı / bitti,
3. **barge-in algılama** — robot konuşurken karşıdan gerçek konuşma gelmesi.

Ayrı bir VAD yazılmaz. Enerji tabanlı VAD zaten bu üçünü de yanlış yapardı:
pencereden gelen trafik yüksek ve ısrarcıdır, ama harmonik değildir ve hece
hızında modüle olmaz (ölçüldü: konuşma 0.59/0.83, araba 0.32/0.03, uğultu
0.99/0.00).

### K4 — `is_robot_speaking` tek kaynaktan yayılır

`AudioOutputManager.is_playing` tek doğruluk kaynağı. Buradan iki yere gider:

- `tracker.step(...)` → `process_raw_doa(is_robot_speaking=True)`. Bu parametre
  gaze çekirdeğinde **zaten var** ve güveni 0.15 ile çarpıyor; standalone hiç
  geçmiyordu. Geçirilmezse robot kendi sesine döner.
- `VoiceLoop` → kendi çıkışını transkribe etmez, ve barge-in penceresini açar.

### K5 — Barge-in `generation_id` üzerinden

`AudioOutputManager` zaten üretim numarası taşıyor ve `interrupt(new_generation_id)`
sunuyor. Yeni bir kesme mekanizması yazılmaz.

## Bileşenler

### `standalone/voice.py` — `VoiceLoop`

Kablolama. Kendi thread'inde döner.

```
VoiceLoop(audio_source, stt=None, llm=None, tts=None, output=None, session=None,
          wake_word="hey astro", ...)
    .start() / .stop()
    .is_speaking -> bool     # AudioOutputManager.is_playing + yankı soğuması
```

`is_speaking`, `AudioOutputManager.is_playing`'i doğrudan yansıtmaz: hoparlör
sustuktan sonra `ECHO_MUTE_COOLDOWN_S` (0.65 s) boyunca `True` kalır. Bayrak
anında düşerse mikrofona hâlâ yolda olan kendi sesi transkribe edilir.

Bütün sağlayıcılar enjekte edilebilir; test sahtelerle koşar, ağ çağrısı olmaz.

Döngü:

1. Halka tampondan pencere çek.
2. `SpeechDetector` ile sözce sınırını izle: konuşma başladı → biriktir; **art
   arda `UTTERANCE_SILENCE_S` (0.8 s) boyunca konuşma görülmezse** sözce kapanır.
   0.8 s, cümle içi duraklamayı (hece arası ≤0.25 s, virgül duraklaması ~0.5 s)
   sözce sonundan ayıracak kadar uzun, tur gecikmesini hissedilir kılmayacak kadar
   kısa. Eşik dışarıdan verilebilir; gerçek kayıtla yeniden ayarlanacak.
   Sözce üst sınırı 10 s (halka tampon boyu).
3. Robot konuşuyorsa: sözce **transkribe edilmez**, yalnızca barge-in kararı için
   kullanılır → `output_manager.interrupt(...)`.
4. Sözce → `STTRouter.transcribe(audio_arr, wav_bytes, 16000)`.
5. `ConversationSession.is_wake_word(text, wake_word)` / oturum açık mı?
   Değilse tur düşürülür.
6. `LlmClient.reply(text)` (aşağıda).
7. `TTSRouter.synthesize_and_play(text, generation_id, output_manager, "tr")`.

Paylaşılan nesneler (yeniden yazılmaz): `STTRouter`, `TTSRouter`,
`AudioOutputManager`, `ConversationSession`.

### `standalone/llm.py` — `LlmClient`

**Yeni kod.** `ai_brain_node` içindeki gömülü çağrıların yerine geçecek, tek
sorumlulukla: metin al, metin döndür.

```
LlmClient(client=None, model=None, system_prompt=None, timeout_s=...)
    .reply(user_text: str) -> Optional[str]
```

- Sağlayıcı dışarıdan enjekte edilir (test sahte istemciyle koşar, ağ yok).
- İstemci yoksa `None` döner — program durmaz, döngü turu düşürür.
- C1'de tek atış; streaming ve persona C2'de eklenecek. C2'nin `system_prompt`'u
  değiştirebilmesi için parametre şimdiden var.

### `standalone/sources.py` — halka tampon

`AudioSource`'a eklenir:

```
.read_window(seconds) -> Optional[np.ndarray]   # mono, yakalama hızında
.sample_rate -> int                             # tamponun gerçek hızı
```

Mevcut `_window` konuşma verdisi için kısa (0.6 s) tutuluyor; sözce için ayrı ve
daha uzun (≈10 s) bir halka tampon gerekir. İkisi karıştırılmaz.

**Örnekleme hızı tuzağı:** tampon 16 kHz garanti edemez. `AudioSource._choose_device`
dizi modunda 16 kHz açıyor ama stereo modda aygıtın kendi hızını kullanıyor
(laptopta 44.1 kHz). STT 16 kHz bekliyor. Bu yüzden tampon **yakalama hızında**
tutulur ve dönüştürme tek bir yerde, STT'ye verilmeden hemen önce yapılır. Hızın
sessizce yanlış varsayılması bu depoda daha önce bedel ödetti (int16 eşiği float32
akışa uygulanmış, kapı her bloğu elemişti); bu yüzden hız tahmin edilmez,
`sample_rate` ile taşınır.

### `standalone/tracker.py` ve `track.py`

- `GazeTracker.step(..., is_robot_speaking: bool = False)` → `process_raw_doa`'ya
  geçirilir.
- `track.py` `VoiceLoop`'u kurar, başlatır, kapanışta durdurur; her çevrimde
  `voice.is_speaking` değerini `tracker.step`'e taşır.

## Bozulma davranışı

Bu program masaüstünde, robotsuz çalışabilmek zorunda; eksik parça durdurmaz,
görünür olur:

| eksik | davranış |
|---|---|
| `openai` paketi / API anahtarı yok | `VoiceLoop` başlamaz, tek satır uyarı basar, gaze normal çalışır |
| hoparlör yok | STT ve LLM çalışır, TTS uyarı basar |
| mikrofon yok | ses döngüsü hiç başlamaz (bugünkü davranışın aynısı) |
| STT bir turu düşürür | tur atlanır, oturum açık kalır, döngü devam eder |

## Test stratejisi

Donanımsız ve ağsız koşan birim testleri:

- sözce sınırı: konuşma başlangıcı/bitişi doğru yerde kapanıyor mu
- wake word kapısı: oturum kapalıyken wake word'süz tur düşüyor, wake word'lü tur
  oturumu açıyor
- yankı bastırma: robot konuşurken gelen sözce transkribe edilmiyor
- barge-in: robot konuşurken gerçek konuşma gelirse `interrupt` çağrılıyor,
  gürültü gelirse çağrılmıyor
- `is_robot_speaking` yayılımı: `tracker.step`'ten `process_raw_doa`'ya ulaşıyor
- bozulma: istemci yokken program çalışmaya devam ediyor

Sahte sağlayıcılar (STT/LLM/TTS) enjekte edilir; ağ çağrısı yapılmaz.

Uçtan uca: C1 bitiminde **bir kez** gerçek OpenAI turu (kullanıcı izin verdi),
laptop mikrofonu ve hoparlörüyle. Sonuç yazıya geçirilir.

Regresyon: mevcut 98 standalone testi yeşil kalmalı.

## Bağımlılıklar

`openai` ve `python-dotenv` **zaten `requirements.in`'de tanımlı**, yalnızca bu
venv'e kurulmamış. Yeni bağımlılık eklenmiyor; venv projenin beyanına eşitleniyor.

## Riskler ve açık sorular

- **Sözce sınırı eşikleri sentetik sesle ayarlanacak.** `SpeechDetector`'ın
  eşikleri henüz gerçek kayıtla doğrulanmadı. Bu oturumda iki hipotez sentetik
  sahnede çürüdüğü için eşikler dışarıdan verilebilir tutulacak ve gerçek kayıt
  gelince yeniden ayarlanacak.
- **Yankı bastırma yalnızca `is_playing` bayrağına dayanıyor.** Hoparlör sesi
  mikrofona gecikmeyle ulaşır; `.env`'deki `ECHO_MUTE_COOLDOWN_S=0.65` bunun için
  var. Bayrağın düşmesinden sonra bir soğuma penceresi uygulanacak.
- **Wake word eşleşmesi `ConversationSession.is_wake_word`'e bırakılıyor**; onun
  Türkçe normalleştirmesi bu tasarımda sorgulanmadı.
- Donanım gelene kadar ReSpeaker kanal sırası açık (bkz. A2); bu C1'i etkilemez,
  çünkü C1 mono karışım kullanır.
