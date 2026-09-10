# ASTRO — sosyal robot kafa/bakış yığını

Konuşana ve görünen kişiye dönen bir robot kafası. ROS 2 Humble, Python düğümler +
Arduino Mega firmware.

## Çalıştırma

```bash
./scripts/build.sh                    # ros2_ws içinden, venv Python'uyla derler
./scripts/build.sh astro_vision       # tek paket
source ros2_ws/install/setup.bash
ros2 launch astro_vision camera.launch.py source:=webcam show_debug:=true
```

Testler **repo venv'iyle** koşar, sistem Python'uyla değil:

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m pytest ros2_ws/src/astro_base/test ros2_ws/src/astro_vision/test -q
```

`install/` bir kopyadır, symlink değil: kaynak değişikliği `./scripts/build.sh`
çalıştırılmadan çalışan sisteme yansımaz. YAML config için de aynısı geçerli.

## Boru hattı

```
kamera 30 Hz → face_detector_node ──/vision/faces──→ social_gaze_node ──/head/command──→ serial_bridge → Arduino
                        │                                    ↑                                  │
                  /vision/face_image                    /audio/doa                     /head/state (encoder)
```

- **Algı 30 Hz, kontrol 50 Hz.** Gaze döngüsü görüntüden bağımsız akar.
- `social_gaze_node` `/gaze/debug`'a her çevrimde JSON teşhis yayınlar —
  `attention_owner`, `visual_valid`, `desired_yaw_deg`, `actual_yaw_deg`. Bir sorunu
  tahmin etmeden önce buraya bak.
- `ros2 run astro_vision session_recorder` koşuyu açıklamalı video olarak kaydeder
  (kutular + gaze durum şeridi). Robot üstünde olup masaüstünde üretilemeyen
  sorunlar için.

## Kritik değerler ve nedenleri

| Değer | Yer | Neden |
|---|---|---|
| Kafa limiti ±85° | `astro_base/config/calibration_params.yaml` | Encoder 440 tick / 170° ile karakterize edildi. Firmware ±180 kabul eder; ölçülmemiş aralığa çıkma. |
| `acquisition 0.75 / hold 0.40` | `gaze/target_manager.py` | Hedef edinmek için 0.75, kilidi korumak için 0.40. Köprülenen (bayat) tespitler kasten 0.75'in altına düşer. |
| `min_confidence 0.50` | `gaze/visual_perception.py` | Altındaki gözlem hiç geçerli sayılmaz. |
| `process_every_n: 1` | `astro_vision/config/camera_params.yaml` | Yüklü kare ~13 ms, 30 Hz bütçesi 33 ms. Büyütmek doğrudan takip gecikmesi demek. |
| Kafa hızı / deadband | `arduino/astro_firmware/src/main.cpp` | Hız limiti **firmware'de**; `social_gaze_node` Arduino'ya ham hedefi yolluyor, planlayıcı çıktısını değil. YAML'daki `max_velocity_deg_s` kafaya ulaşmaz. Deadband 3 tick (1.16°) çünkü dişli boşluğu 0.85°. |
| Akustik zarf 75° / 121° | `gaze/audio_perception.py` | 75° sohbet konisi anında geçer. 75–121° arası ısrar ister (yankıyı insandan ayırmak için). 121° = kafa limiti 85 + kamera yarı-FOV 36; ötesi kadraja giremez, gövde dönüşü ister (yok). |
| DDS SHM profili | `astro_vision/config/fastdds_shm.xml` | 900 KB'lık görüntü topic'i, Linux'un 208 KB UDP tamponuna sığmıyor. Profil olmadan %70 kare kaybı. `camera.launch.py` bunu set eder. |

## Bu kod tabanında tekrar eden tuzaklar

- **`msg.data = <bytes>`**: rclpy `uint8[]` alanına `bytes` verilince 921.600 elemanı
  saf Python'da iki kez doğruluyor — kare başına 45 ms. `array.array("B", ...)` hızlı
  yola girer (`image_utils.bgr_to_imgmsg`).
- **Büyük topic'ler sessizce kaybolur.** `/oak/rgb/image_raw` ve `/vision/face_image`
  SHM profili olmadan onda bir hızda akar ve kimse hata vermez. Bir topic'i dinleyen
  yeni bir araç yazarken profili eşleştir.
- **Config drift**: YAML'da tanımlı ama node'un `declare_parameter` etmediği anahtarlar
  ROS tarafından sessizce yutulur. `test_capture_and_detection_rate.py` bunu yakalayan
  bir test içeriyor; yeni parametrede aynı korumayı kur.
- **Kafa geri beslemesi yoksa takip sessizce ölür.** Bütün kerterizler
  `actual_head_yaw + kamera_açısı`; encoder susarsa hedef merkeze çöker. Node artık
  bir kez uyarıyor.
- **İki firmware kopyası var, biri bayat.** Kanonik: `astro_firmware/src/main.cpp`
  (PlatformIO, robota yüklenen). `AstroFirmware/AstroFirmware.ino` ondan kopmuş —
  tick ölçeği yanlış (1.5 vs 2.5882) ve hız rampası yok. Yüklemeyin.
- **Ses nereye değil, kim'e karar verir.** Yüz görünürken kerteriz yalnızca
  görüntüden gelir; füzyon ikisini ortalarsa DOA gürültüsü hedefi deadband'in
  ötesine itip kafayı boşuna oynatıyordu (ölçüldü: 5.1° çekme).
- **Headless test**: node'lar rclpy import edilemediğinde `ros_compat.py`'deki mock
  Node'a düşer; `test/conftest.py` rclpy'yi bilerek bloklar. Node seviyesinde test
  yazmak bu sayede mümkün.

## Çalışma tarzı

- Ölçmeden değiştirme. Bu depoda "yavaş" görünen şeylerin çoğu bir throttle ya da
  bir taşıma sorunuydu, algoritma değil.
- Bir performans/davranış iddiası yaparken sayıyı nereden aldığını yaz. Sentetik test
  sahnesi kurarken sahnenin kendisinin doğru olduğunu ayrıca doğrula.
- `feat/social-gaze-audiovisual-tracking` paylaşılan daldır ve birden fazla kişi
  çalışır. Geçmişi `reset --hard` + force-push ile geri alma; `git revert` kullan.
