# ROS'suz takip

Kamerayı, mikrofonu ve Arduino'yu doğrudan açan tek process'lik bir takip programı.
ROS yok: DDS yok, topic yok, launch yok, tek log, çalışan debugger.

```bash
./.venv/bin/python standalone/track.py                      # kamera + ses
./.venv/bin/python standalone/track.py --serial /dev/ttyACM0   # + kafa
./.venv/bin/python standalone/track.py --no-window --seconds 30
./.venv/bin/python standalone/track.py --no-voice          # yalnizca takip
./.venv/bin/python standalone/track.py --no-voice --fixed-head  # sabit dizüstü sensörleri
```

Çıkmak için pencerede `q` ya da `Esc`.

## Dizüstünde ses yönünü ölçmek

Kamera ve mikrofon masada sabitken `--fixed-head` kullanın. Bu mod kafa referansını
0° tutar. Normal açık çevrim yolu komutun uygulandığını tahmin eder; sensörler
fiziksel olarak dönmeyince aynı yüz açısı tahmine tekrar eklenir ve hedef limite
sürüklenebilir. `--fixed-head` ile `--serial` birlikte kullanılamaz.

```bash
./.venv/bin/python standalone/track.py --no-voice --fixed-head --log-interval 0.5
```

Logda `gercek` encoder ölçümü, `tahmin` açık çevrim hesabı, `sabit` ise masaüstü
referansıdır. `kafa:X` encoder olmadığını belirtmeye devam eder. Sabit referans,
robotun dönüşünü veya ReSpeaker montajını doğrulamaz.

Ölçümde önce kısa bir sessizlik bırakın, sonra laptopun solundan ve sağından
normal cümleler söyleyin; hangi zaman aralığında nerede olduğunuzu kaydedin.
Uzun “aaa” hece modülasyonu taşımadığı için konuşma filtresinden elenebilir.
`rms`, `harm` ve `mod` sütunları ses düzeyini, harmonikliği ve hece modülasyonunu
gösterir; yön bulunamadığında da konuşma penceresinin ölçüleri görünür.
Yalnız sese göre hedef seçimini sınarken yüz kadraj dışında olmalı; görünür
hedefin yeni konuşana geçişi engellemesi ayrıca ele alınacak bir karar kuralıdır.

## Neden var

Bu yığındaki hataların çoğu algoritmada değil taşımadaydı: DDS 900 KB'lık görüntü
topic'inin %70'ini düşürüyordu, YAML anahtarları sessizce yutuluyordu, bir `bytes`
ataması kare başına 45 ms yiyordu. Bunların hiçbiri burada yok.

Bu yüzden program bir soruyu doğrudan cevaplıyor:

> Burada takip temizse → sorun **boru hattında**.
> Burada da aynıysa → sorun **algoritmada ya da ayarda**.

## Beyin paylaşılıyor, kopyalanmıyor

Karar veren her şey — `VisualPerceptionCore`, `VisualTrackerCore`,
`AudioVisualFusionCore`, `TargetManagerCore`, `SocialGazeFSM`, `MotionPlannerCore` —
ROS düğümünün kullandığı nesnelerin **aynısıdır**, `astro_base.gaze` içinden import
edilir. Burada yerel olan tek şey aralarındaki kırk satırlık kablolama.

İkinci bir kopya yazılmamasının sebebi somut: bu depoda iki firmware kopyası
535 satırın 147'sinde ayrıştı ve bayat olan yüklenirse her açıyı 1.73 kat büyük
uyguluyordu. Elle bakılan ikinci bir kopya kaçınılmaz olarak kayar — ve kaydığında
bu programın cevapladığı soru anlamsızlaşır, çünkü artık aynı beyni ölçmüyor olur.

## Ekrandaki şerit

```
HOLDING_ATTENTION  owner=VISUAL_TRACKING  hedef=person_1  conf=0.92
istenen +6.5  ->  gercek +0.0   [29 fps  ses:V  kafa:X]
```

Hangi katmanın sustuğunu tahmin etmeden okumak için:

| gördüğün | sorun nerede |
|---|---|
| kutu hiç yok | algılama |
| kutu var ama `owner=IDLE` | arbitrasyon — gaze hedefi kabul etmiyor |
| `istenen` değişiyor, `gercek` sabit | aktüatör ya da seri hat |
| `kafa:X` | encoder konuşmuyor, açı komuttan tahmin ediliyor |

## Donanım zorunlu değil

Kamera dışında hiçbiri şart. ReSpeaker yoksa yalnız görüntüyle takip eder; Arduino
yoksa açık çevrim çalışır ve kafa açısını komuttan tahmin eder (bunu da çıkışta
söyler). Masaüstünde, robot olmadan çalışması bilinçli: yalnızca bitmiş robotta
çalışan bir teşhis aracının teşhis değeri az olur.

Sesli yanıt `OPENAI_API_KEY` ister. Anahtar yoksa program tek satır uyarı basar
ve yalnızca takip yapar — konuşma, görmenin önkoşulu değil.

## Testler

```bash
./.venv/bin/python -m pytest standalone/test -q
```

Testler gerçek robot istemez. Seri protokol (çerçeveleme, CRC, encoder
telemetrisi, heartbeat), kaynakların donanımsız davranışı ve takipçinin
ROS tarafıyla aynı garantileri koruduğu kapsanıyor — kafa açısına göre kerteriz,
encoder susunca merkeze çökmeme, ve sesin nişanı çekmemesi.

## Teşhis şeridindeki ses sütunu

`--log-interval` satırının sonunda, kerteriz varsa bir sütun daha basılır --
kafayı yalnızca konuşma çevirebildiği için "kerteriz var ama kafa dönmüyor"
sorusunun sesle ilgili yarısını buradan okumak, tahmin etmekten iyidir:

| gördüğün | anlamı |
|---|---|
| `[konusma 0.82]` | pencere konuşma sayıldı, doluluk (confidence) yanında |
| `[elendi: <sebep>]` | pencere konuşma sayılmadı, `speech.reason` sebebi |
| `[pencere yok]` | kerteriz var ama konuşma penceresi henüz dolmadı |
