# ReSpeaker yön araştırması — 8 Eylül 2026

## Kapsam ve belirsizlik

Bu not Seeed belgeleri ve ReSpeaker'ın resmi kaynak koduna dayanır. Robottaki ürünün tam modeli, firmware sürümü, mikrofon kanal sırası ve montaj eksenleri **doğrulanmadı**. ReSpeaker donanımı okunmadı, mikrofon kaydı alınmadı. İlk araştırma turunda üretim kodu/config/firmware değiştirilmedi; kullanıcının sonraki laptop loguyla yapılan teşhis düzeltmesi aşağıda ayrıca anlatılır. Önerilen donanım ölçümleri henüz sonuç değildir.

Kullanıcının son açıklaması: şu anda ReSpeaker bağlı değil, yalnızca dizüstünün iki kanallı mikrofonu mevcut; çalışma tercihi ROS kullanmadan ilerlemek. Robottaki mikrofon kafayla birlikte dönüyor, R2D2 benzeri göz konumunda eğik duruyor. Kafa mekanik olarak sürekli dönebiliyor, mevcut yazılım aralığı sınırlı. Bunlar kullanıcı beyanıdır; bu araştırmada donanım ölçümüyle doğrulanmadı.

## Önce ürün ayrımı

- **USB Mic Array v2.0 / XVF3000:** USB ses kartı ve kart üzerinde DOA/VAD işleme içerir. Belgelenmiş 6 kanallı firmware'de sıfır tabanlı sıra: **0 işlenmiş ASR, 1–4 ham mikrofon, 5 playback birleşimi**. Tek kanallı firmware yalnızca işlenmiş ASR verir. Dolayısıyla USB akışının ilk dört kanalını dört ham mikrofon kabul etmek yanlıştır. Yazılımsal TDOA için bu modelde dört ham kanalın doğru sırayla seçilmesi gerekir. Bu eşleme diğer modellere taşınmamalıdır. [Seeed v2.0, firmware bölümü](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/#update-firmware)
- **4-Mic Array for Raspberry Pi:** 40 pin üzerinden bağlanan, AC108 ADC ve I2S/TDM kullanan Pi genişleme kartıdır. Belgelenen DOA/VAD yazılım algoritmaları Pi üzerinde çalışır; USB XVF3000 kontrol register'larına sahip olduğu varsayılamaz. [Seeed Pi 4-Mic](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/)
- **4-Mic Linear Array Kit:** Ayrı mikrofon çubuğu ve HAT içerir. Sekiz girişin ilk dördü geçerli mikrofon verisi; sonraki ikisi kullanılmayan giriş, son ikisi playback echo kanalıdır. Dairesel dizi geometrisi bu ürüne uymaz. [Seeed Linear Array](https://wiki.seeedstudio.com/ReSpeaker_4-Mic_Linear_Array_Kit_for_Raspberry_Pi/)
- **XVF3800 USB 4-Mic:** Daha yeni ve farklı kontrol arayüzü olan başka bir olasılıktır. Seeed varsayılan USB kimliğini `2886:001a` verir; araç/sürüme göre `DOA_VALUE` veya `AEC_AZIMUTH_VALUES` okunur. Komutların birimi ve veri yapısı kendi sürümünün komut açıklamasından doğrulanmalıdır. XVF3000 `tuning.py` varsayılanı ise `2886:0018`'dir. [Seeed XVF3800 başlangıç](https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/), [XVF3000 resmi tuning.py](https://github.com/respeaker/usb_4_mic_array/blob/master/tuning.py)

## USB DSP yönü bağımsız bir teşhis yolu sağlar

XVF3000 resmi `tuning.py`, USB vendor control transfer ile `DOAANGLE` okur; değer aralığı 0–359'dur. `direction` bu değeri, `is_voice()` ise `VOICEACTIVITY` değerini döndürür. `SPEECHDETECTED` ayrıca tanımlıdır; bunları aynı değişken saymamak gerekir. Kaynak, yön referansının firmware derleme ayarına bağlı olduğunu açıkça belirtir: “Orientation depends on build configuration.” Bu nedenle 0°'nin robot önü olduğu veya pozitif açının robotun soluna döndüğü varsayılmamalıdır. [Resmi tuning.py](https://github.com/respeaker/usb_4_mic_array/blob/master/tuning.py)

Seeed'in DOA/VAD örnekleri PCM akışını açmadan kontrol değerlerini okur. Model doğrulanınca bu yol, yazılımsal kanal seçimi ve GCC-PHAT'ten bağımsız karşılaştırma sağlayabilir. Örnekler eski Python sürümüne aittir; doğrudan çalışacakları doğrulanmış değildir. Başlangıçta firmware veya VAD eşiği değiştirmek yerine mevcut değerler gözlenmelidir. [Seeed DOA/VAD örnekleri](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/#doa-direction-of-arrival)

Resmi Pi örneği dört kanalda `[0,2]` ve `[1,3]` çiftlerini, `0.08127 m` çift mesafesini ve sonrasında sabit açı dönüşümlerini kullanır. Bunlar o örneğin kanal/geometri varsayımlarıdır; USB ürüne kopyalanacak evrensel sabitler değildir. [Resmi mic_array.py](https://github.com/respeaker/mic_array/blob/master/mic_array.py)

## Kodda doğrulanan USB protokol uyuşmazlığı

Resmi `PARAMETERS` sözlüğü modül/offset çiftlerini `SPEECHDETECTED=(19,22)`, `VOICEACTIVITY=(19,32)`, `DOAANGLE=(21,0)` olarak tanımlar. `Tuning.read()` tamsayıda `cmd=0x80|offset|0x40` üretir; transfer sırası `(0xC0, 0, cmd, module_id, 8, timeout)` olur. Dolayısıyla **wValue/wIndex**: speech `0xD6/19`, voice `0xE0/19`, DOA `0xC0/21`. İlk `0xC0` USB istek türüdür; DOA'nın wValue alanındaki aynı sayıdan ayrı bir alandır. Tamsayı cevapta ilk 32 bit kullanılır. [Resmi ham kaynak: PARAMETERS ve Tuning.read](https://raw.githubusercontent.com/respeaker/usb_4_mic_array/master/tuning.py)

Yerel `audio_stream_node.py` satır 180–236'da `ReSpeakerHID._read_param()` ise `(0xC0, 0, param_id, 0, 8, timeout)` gönderiyor. `param_id=19/21` değerleri yanlış alana konuyor; modül alanı sıfır, offset ve okuma/tamsayı bitleri eksik. Bu, **XVF3000 resmi protokolüyle kaynak kod düzeyinde kesin uyuşmazlıktır**. Sınıf adı HID olsa da kullanılan aktarım vendor control transfer'dir. [Yerel uygulama](../ros2_ws/src/astro_audio/astro_audio/audio_stream_node.py)

Bu bulgu, USB VAD/DOA okumalarının neden başarısız olabileceğini açıklar; cihaz bağlı olmadığından sahadaki arızanın nedeni olduğu henüz kanıtlanmadı. Laptopun iki kanalı, ReSpeaker dört ham kanal geometrisinin veya USB kontrolünün doğrulaması değildir. Üretim düzeltmesi yapılmadan, sahte USB aygıtıyla beklenen aktarım alanları test edilebilir; gerçek cihaz testi daha sonra gerekir.

Kapsam ayrımı: `standalone/sources.py` bu USB sınıfını kullanmaz. Ayrıca mevcut ROS PCM yolu geçerli GCC-PHAT sonucunda kendisi `/audio/vad=True` yayınlar. USB bulgusu bütün ses yollarının sustuğunu veya tek kök neden bulunduğunu göstermez.

## Bu oturumda yerel doğrulama ve seçilen davranış

İncelenen HEAD: `836a970e`, dal: `standalone/ros-free-tracker`. `lsusb` listesinde ReSpeaker yok; `/proc/asound/cards` yalnız NVIDIA ve Intel PCH gösterdi. Sandbox dışındaki `sounddevice.query_devices()` sorgusunda gerçek giriş aygıtı **4: HDA Intel PCH: ALC294 Analog (hw:1,0), 2 giriş kanalı** çıktı. Aygıt numarası yeniden bağlama/başlatmada değişebilir. Mikrofon kaydı alınmadı.

`4ff46916` commit'i `astro_bringup/astro_social_gaze.launch.py` içine ROS ses ve Realtime düğümlerini bağlamış. ROS'suz giriş hâlâ `standalone/track.py`. Yerel takip denemesinin başlangıcı, depo kökünde `./.venv/bin/python standalone/track.py --no-voice`; gerekirse güncel aygıt numarası `--audio-device` ile seçilir. `--no-voice` ses yönü takibini kapatmaz, sağlayıcı/STT/LLM/TTS turunu kapatır. Bu oturumda canlı takip başlatılmadı. Sonraki adımda kullanıcıyla birlikte kontrollü sessizlik/sol/sağ konuşma ölçümü yapılacak.

**Kullanıcının seçtiği davranış:** Kadraj dışında yeni biri konuşursa ona dön; yüzünü bulunca hassas yönü görüntüden al. Görüntünün nişanı belirlemesi ile görünür kişinin hedef değiştirmeyi engellemesi ayrı kurallardır.

Karar katmanı bu tercih için iki karşılaştırmalı girdide çalıştırıldı. `GazeTracker` ve gerçek ortak sınıflar kullanıldı; mikrofon dalga biçimi taklit edilmedi. Sahne: 640×480 kare, yüz kutusu `(270,150,100,100)`, güven 0.95, sabit encoder 0°, ilk 20 çevrim sessiz, sonraki 60 çevrim ham DOA 60° ve `SpeechVerdict(is_speech=True, confidence=0.95)`, çevrim zaman adımı 0.05 s. Sahnenin kontrolü: kutu merkezi x=320; 60° mevcut yarı-FOV 36° dışında; ortak kalibrasyon ham 60° ve kafa 0° için gövde açısını −60° veriyor. Sonuçlar:

- Yüz yok: `ACTIVE_SPEAKER`, hedef **−60°**.
- Merkezde görünür yüz var: `VISUAL_TRACKING`, hedef **0°**.

Bu karşılaştırma, aynı kabul edilen ses girdisinin görünür hedef nedeniyle seçilemediğini gösterir. Gerçek konuşma sınıflandırma doğruluğu, akustik açı hatası, tepki süresi veya motor başarısı ölçülmedi. İlgili kararlar `gaze/sensor_fusion.py`, `gaze/target_manager.py` ve durum makinesinde; değişiklik ortak beyinde yapılmalı.

Başlangıç test sonucu: `./.venv/bin/python -m pytest standalone/test -q -o faulthandler_timeout=30` → **164 passed, 7 subtests passed in 4.42s**. Sandbox içindeki ilk koşu `test_without_a_microphone_there_is_simply_no_bearing` testinin gerçek `sounddevice` başlatmasına girmesiyle takıldı; yığın izi alınarak durduruldu, aynı repo venv'iyle sandbox dışında tamamlandı. Bu test sonucu donanım kabulü değildir. Tüm depo testleri ve ROS derlemesi bu araştırma turunda çalıştırılmadı; üretim kaynakları değişmedi.

## Kullanıcının laptop koşusu ve sabit referans düzeltmesi

Kullanıcı `track.py --no-voice` çıktısını paylaştı: ALC294 stereo @44100 Hz,
Arduino yok, 736 kare / 28.9 s = 25.5 Hz. Yön değerleri ve bazı konuşma kabulleri
var; hedef kimliği bütün paylaşılan satırlarda `person_1`. Kullanıcı açıları
iki mikrofona göre kabaca doğru bulduğunu, normal cümleler yerine sürekli “aaa”
dediğini belirtti. Bu nedenle “harmonik ama hece modülasyonu yok” retleri gerçek
cümlelerde yanlış ret oranı olarak yorumlanamaz. Logdaki `ACTIVE_SPEAKER` etiketi
tek başına yeni konuşana geçildiğini göstermez.

Sabit sensör koşulu ayrıca yeniden üretildi: 640×480 karede kutu
`(330,160,100,100)`, merkez x=380, güven 0.95, ses yok; ortak takipçi 200 çevrim
boyunca 30 Hz zaman adımıyla beslendi. Encoder girdisi yokken hedef −6.2°'den
−85°'ye birikti; her çevrimde bilinen sabit kafa referansı 0° verilince hedef
−6.2° kaldı. Bu, donanımsız karar denemesidir; gerçek motor hareketi ölçülmedi.

`standalone/track.py` artık açıkça seçilen `--fixed-head` seçeneğini destekler.
Sabit kamera/mikrofonun referansı ortak takipçiye 0° olarak aktarılır; encoder
bağlı gösterilmez (`kafa:X`). Seçenek `--serial` ile birlikte kullanılamaz.
Normal robot/açık çevrim yolu korunur. Log ve ekran `gercek` / `tahmin` / `sabit`
etiketlerini ayırır. Konuşma penceresinin `rms`, `harm`, `mod` ölçüleri yön
bulunmadığında da gösterilir. Konuşma eşikleri ve ortak hedef seçim kuralları
bu düzeltmede değiştirilmedi.

Sonraki kontrollü koşu:
`./.venv/bin/python standalone/track.py --no-voice --fixed-head --log-interval 0.5 --seconds 30`.
İlk 5 s sessizlik; 5–15 s laptopun solundan, 15–25 s sağından normal cümleler;
son 5 s sessizlik. Yalnız ses hedefini sınamak için yüz kadraj dışında tutulmalı.
ReSpeaker montajı ve tam çevre takibi bu denemenin kapsamı dışında kalır.

Düzeltme sonrası aynı repo venv'iyle standalone paketi: **168 passed,
7 subtests passed in 3.43s**. Yeni CLI/sabit sahne ve log kontrollerinin 17 testlik
alt kümesi de geçti. Kod incelemesinde ek düzeltme gerektiren bulgu çıkmadı.
Gerçek mikrofonla bu yeni mod henüz çalıştırılmadı; sonraki kullanıcı koşusu bunu
doğrulayacak. ROS kaynakları değişmediği için bu değişiklik için ROS derlemesi
gerekmez.

## Devam oturumundaki gerçek laptop ölçümleri

`75d5b3db` üzerindeki sabit referans yolu gerçek webcam ve ALC294 stereo girişle
30 s çalıştırıldı; log `/tmp/astro-ses-testi.log` içinde. Çıkış özeti 850 kare /
30.1 s = 28.2 Hz. Yüz bütün 57 durum satırında görünür kaldı. Geçiş mesajları
geciktiğinden ve kullanıcının fiili konuşma/yön zamanları doğrulanmadığından koşu
kontrollü sol–sağ denemesi sayılmaz.

Log örnekleri: 7 satırda yön, 4 satırda konuşma kabulü, 0 satırda ikisi birlikte.
Bunlar bütün ses bloklarının sayımı değildir; yanlış ret veya yön doğruluk oranı
hesaplanamaz. Sabit kafa referansı bütün satırlarda 0°, hedef +8.5° kaldı.

ALSA okuması `Capture` için iki kanalda +30 dB ve `Internal Mic Boost` için iki
kanalda +30 dB gösterdi. Aynı aygıttan, aynı 44100 Hz/iki kanal/float32 biçiminde
ayrı 3 s örnek alındı. Tam skalaya yakın (`abs(sample) >= 0.999`) örnek oranları
sol %32.938, sağ %33.135; tepe iki kanalda 1.0; RMS 0.62427/0.63713.
Bu örnekte belirgin kırpılma var. Paylaşılan `StereoDOA` kestiricisi 4096 örneklik
32 bloğun 23'ünde yön döndürdü; keskinlik min/medyan/max 1.736/4.043/16.934.
Kaynak yönü etiketli olmadığından bunlar doğru yön sayısı değildir.

`Internal Mic Boost` geçici olarak 0 dB'ye indirilerek ayrı 3 s örnek alındı:
tam skalaya yakın örnek oranı %9.846/%10.114; RMS 0.35924/0.36124; tepe 1.0/1.0.
32 bloğun 15'i yön döndürdü, keskinlik 1.683/2.854/16.934. İki kayıtta aynı
akustik uyaran sağlanmadı; aradaki farktan iyileşme veya nedensellik çıkarılamaz.
0 dB örneğinde de kırpılma sürdü. Önceki +30 dB boost ayarı `finally` içinde
geri yüklendi ve `amixer` çıktısıyla doğrulandı. Ham ses dosyası saklanmadı.

Sonraki adım kullanıcıdan bu sırada konuşma veya hoparlör/müzik olup olmadığını
netleştirmek; ardından kullanıcı tarafından başlatılan, zamanları etiketli bir
normal konuşma denemesiyle giriş seviyesini ve yön ölçümünü ayırmak. Konuşma
eşikleri, yön kestiricisi ve hedef seçim kodu bu ölçüm turunda değiştirilmedi.

## 30° montaj açısı tek başına yeterli bilgi değil

Aşağıdakiler geometrik çıkarımdır, üreticinin bu robot için ölçtüğü sonuçlar değildir. Düzlemsel dizide uzak alan gecikmesi `c·τij = (ri−rj)·u` ile modellenir. Tüm mikrofonlar yerel `z=0` düzlemindeyse gecikmeler `u_z` işaretini ayırt edemez: düzlemin iki tarafındaki ayna yönler aynı gecikmeleri üretebilir. Bu, “düzlemsel dizi hiçbir yükseklik bilgisi taşımaz” demek değildir; tam 3B yönün bu ölçümden tekil biçimde çıkarılamayacağı anlamına gelir. Seeed'in XVF3800 geometri örneğinde de dört mikrofonun z koordinatı sıfırdır; bu örnek robottaki ürünün ölçüsü olarak kullanılmamalıdır. [Seeed geometri sorgusu](https://wiki.seeedstudio.com/respeaker_xvf3800_introduction/)

Yalnızca yatay düzlemde 30° döndürülmüş kart için işaret/ofset kalibrasyonu yeterli olabilir. Kart düzlemi pitch veya roll ile eğilmişse dizi düzlemindeki açı dünya yatay açısı değildir; hata kaynak yüksekliğine de bağlı olabilir. Firmware'in tek açısını 3B vektör sayıp rastgele rotasyon uygulamak eksik bilgiyi tamamlamaz. Önce kart düzlemi, ses deliklerinin baktığı taraf, robot önü ve eğimin hangi eksende ölçüldüğü belirlenmelidir. Kaynağın göz hizasında olması bu eksenleri tek başına belirlemez.

Mikrofon kafayla döndüğünden kalibre edilmiş kafa-göreli yönün gövde yönüne dönüşümü aynı zamandaki gerçek kafa açısına dayanmalıdır. Yalnızca yatay dönüş varsayımı geçerliyken `gövde_kerterizi = wrap(gerçek_kafa_yaw + kalibre_edilmiş_yerel_yön)` kullanılabilir. Sabit konuşmacıyla kafa döndüğünde gövde kerterizinin sabit kalması, işaret/ofset ve zaman eşleşmesi için yararlı kontroldür. Bu formül eğik dizinin eksik 3B bilgisini çözmez.

## Ölçüm önerisi ve karar sırası

1. **Kimlik:** Ürün etiketi/çip, bağlantı türü, USB VID:PID, ALSA adı, mevcut firmware ve açılan giriş kanal sayısı birlikte kaydedilsin. “ReSpeaker 4-Mic” tek başına seçim ölçütü olmasın.
2. **Sabit kafa:** Robot önünde, solunda ve sağında bilinen açılarda tek konuşmacı ile ham DSP DOA/VAD, yazılımsal DOA/güven ve zaman damgaları karşılaştırılsın. Aynı açılar farklı kaynak yüksekliklerinde tekrarlansın. İlk etapta motor komutu gerekmiyor.
3. **Kanal/geometri:** Yazılımsal DOA kullanılacaksa gerçek ham kanallar, mikrofon sırası ve çift mesafeleri ürün üzerinden doğrulansın. USB işlenmiş kanalın korelasyona karışması veya yanlış ALSA cihazının seçilmesi kontrol edilsin.
4. **Dönüşüm:** Ön/sol/sağ ölçümleriyle yön işareti ve ofset, sonra diğer açılarla kalan hata ölçülsün. Sadece öndeki tek nokta işareti doğrulayamaz. 359°/0° çevresindeki hata dairesel farkla hesaplansın.
5. **Füzyon, ROS gerektirmeden:** Standalone akışında ses yönü/güveni/VAD, kabul-ret nedeni, hedef kafa açısı ve gerçek encoder açısı aynı zaman damgasıyla karşılaştırılsın. Ses geçerli mi, engelleniyor mu, hedef oluşuyor mu, gerçek kafa geri bildirimi geliyor mu ayrı ayrı belirlenmeli. Mevcut ROS eşdeğerleri `/audio/doa`, `/gaze/debug`, `/head/state`; araştırma bunları çalıştırmayı şart koşmaz. Yüz görünürken sesin “kim” seçiminde kullanılması projenin belgelenmiş tasarımıdır; tek başına ses yönünde dönmemek arıza kanıtı değildir. [Proje kuralları](../CLAUDE.md)
6. **Mekanik ve yazılım ayrı:** Kullanıcı sürekli mekanik dönüş bildirdi. `CLAUDE.md` mevcut encoder kalibrasyonunu 440 tick/170° ve çalışma sınırını ±85° olarak belgeliyor. Sürekli hareket için sarılmış/sürekli açı gösterimi ve tam tur encoder ölçeği ayrıca doğrulanmalıdır. Ölçülmemiş bölgeyi açmak bu araştırmanın sonucu değildir. [Proje kuralları](../CLAUDE.md)

Başarı ölçütü yalnızca “ses duyuldu” değil: bilinen konum → doğru ham yön → doğru robot kerterizi → açıklanabilir kabul/ret → erişilebilir hedef zincirinin aynı zaman çizgisinde doğrulanmasıdır.
