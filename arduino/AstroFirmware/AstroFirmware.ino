/*
 * AstroFirmware - Arduino IDE sürümü
 * ----------------------------------
 * ⚠️  BU DOSYA GÜNCEL DEĞİL — ROBOTA BUNU YÜKLEMEYİN.
 *
 * Kanonik firmware: arduino/astro_firmware/src/main.cpp (PlatformIO).
 * Bu kopya ondan koptu ve 535 satırın 147'sinde ayrışmış durumda. Kafa
 * davranışını bozan iki fark:
 *
 *   1) HEAD_TICKS_PER_DEG burada 1.5000, kanonikte 2.5882 (kalibrasyonla
 *      doğrulanmış: 440 tick / 170°). Bu kopya yüklenirse her açı 1.73 kat
 *      büyük uygulanır — +20° komut ~+35° dönüş demektir.
 *   2) headControl() burada hız rampası içermiyor; kanonikte hedefe 20 °/s
 *      ile rampalanır. Rampa olmadan her komut tam hızlı basamak yanıtıdır.
 *
 * Arduino IDE yolu gerekiyorsa bu dosya main.cpp'den yeniden üretilmeli;
 * elle bakılan ikinci bir motor kontrol kopyası kaçınılmaz olarak tekrar kayar.
 *
 * Kart      : Arduino Mega 2560
 * Seri hız  : 115200 baud (Serial Monitor da 115200 olmalı)
 * Kütüphane : gerekmiyor (sadece Arduino core)
 *
 * - İki tekerlek: BTS7960 + enkoder, 50 Hz hız PID'i (RPM)
 * - Kafa: BTS7960 + enkoderli DC motor, 50 Hz konum PID'i (derece)
 * - Host ile ikili paket protokolü (bkz. protocol.h)
 *
 * IMU YOK: 20/21 (I2C) kafa enkoderine verildi, bkz. pins.h.
 * TMC2209 YOK: kafa artık DC motor, Serial1 sağ enkodere verildi.
 */

#include <Arduino.h>
#include <avr/wdt.h>

#include "pins.h"
#include "protocol.h"

// ====== Parametreler ======
static constexpr uint32_t SERIAL_BAUD = 115200; // CH340/CH341 ve Linux Kernel stabil standart baud
static constexpr float CONTROL_HZ = 50.0f;
static constexpr uint32_t CONTROL_DT_MS = (uint32_t)(1000.0f / CONTROL_HZ);

static constexpr int32_t TICKS_PER_REV_L = 2048; // enkoder CPR*4 uygun biçimde ayarlayın
static constexpr int32_t TICKS_PER_REV_R = 2048;

static constexpr float WHEEL_R_L = 0.06f; // metre (örnek 60mm)
static constexpr float WHEEL_R_R = 0.06f;

static constexpr float KP = 0.6f, KI = 0.2f, KD = 0.0f; // 50 Hz PID için örnek
static constexpr int PWM_MAX = 255;
static constexpr float PID_INTEGRAL_LIMIT = 50.0f; // ✅ FIX: Daha dar anti-windup limit

// Kalibre Edildi: 45 tick / 30 derece = 1.5000 tick/derece (540 tick / 360 derece)
static constexpr float HEAD_TICKS_PER_DEG = 1.5000f;





// Yazılımsal açı limitleri (limit switch yok; açılıştaki konum 0° kabul edilir)
static constexpr float HEAD_MIN_DEG = -180.0f;
static constexpr float HEAD_MAX_DEG =  180.0f;


// Kafa motoru PWM limitleri ve statik sürtünme eşiği
static constexpr int HEAD_PWM_LIMIT = 160;
static constexpr int HEAD_PWM_MIN = 70;

static constexpr float HEAD_KP = 4.0f, HEAD_KD = 0.05f;
static constexpr int32_t HEAD_DEADBAND_TICKS = 1;  // 1 tick ~= 0.78 derece
static constexpr uint32_t HEAD_STALL_MS = 1500;    // PWM'e rağmen tick değişmiyorsa kes (1.5s güvenli süre)


// ====== Diagnostik bayraklari ======
static constexpr uint32_t FLAG_WATCHDOG_TIMEOUT = 0x01;
static constexpr uint32_t FLAG_RESERVED_IMU     = 0x02; // eski IMU_READ_FAIL, artik kullanilmiyor
static constexpr uint32_t FLAG_HEAD_STALL       = 0x04;
static constexpr uint32_t FLAG_HEAD_LIMIT       = 0x08;

// ====== Global Durum ======
volatile int32_t g_left_ticks = 0;
volatile int32_t g_right_ticks = 0;
volatile int32_t g_head_ticks = 0;

static int32_t g_left_last_ticks = 0;
static int32_t g_right_last_ticks = 0;

static float g_left_target_rpm = 0.0f;
static float g_right_target_rpm = 0.0f;

static float g_left_err_i = 0.0f, g_right_err_i = 0.0f;
static float g_left_err_prev = 0.0f, g_right_err_prev = 0.0f;

// Kafa konum kontrolu
static int32_t g_head_target_ticks = 0;
static int32_t g_head_err_prev = 0;
static int g_head_pwm = 0;
static int32_t g_head_stall_ref = 0;
static uint32_t g_head_stall_ms = 0;

static uint32_t g_last_control_ms = 0;
static uint32_t g_last_heartbeat_ms = 0;
static uint32_t g_last_diag_serial2_ms = 0;

// ====== Forensic Diagnostik Sayaçları ======
static volatile uint32_t g_hb_rx_count = 0;
static volatile uint32_t g_hb_ack_tx_count = 0;

static bool g_motors_enabled = true;
static bool g_head_active = false;
static uint32_t g_diag_flags = 0;


// ====== Yardımcılar ======
inline void setMotorPWM(int pwm_fwd_pin, int pwm_rev_pin, int val, int limit) {
  val = constrain(val, -limit, limit);
  if (val >= 0) {
    analogWrite(pwm_rev_pin, 0);
    analogWrite(pwm_fwd_pin, val);
  } else {
    analogWrite(pwm_fwd_pin, 0);
    analogWrite(pwm_rev_pin, -val);
  }
}

inline void setLeftPWM(int v)  { setMotorPWM(L_MOTOR_PWM_FWD, L_MOTOR_PWM_REV, v, PWM_MAX); }
inline void setRightPWM(int v) { setMotorPWM(R_MOTOR_PWM_FWD, R_MOTOR_PWM_REV, v, PWM_MAX); }

inline void setHeadPWM(int v) {
  g_head_pwm = constrain(v, -HEAD_PWM_LIMIT, HEAD_PWM_LIMIT);
  setMotorPWM(HEAD_MOTOR_PWM_FWD, HEAD_MOTOR_PWM_REV, g_head_pwm, HEAD_PWM_LIMIT);
}

inline int32_t readTicks(volatile int32_t& src) {
  int32_t v;
  noInterrupts();
  v = src;
  interrupts();
  return v;
}

void leftEncA()  { g_left_ticks  += digitalRead(L_ENC_B)    ? -1 : +1; }
void rightEncA() { g_right_ticks += digitalRead(R_ENC_B)    ? -1 : +1; }
volatile int8_t g_head_last_dir = 1;

void headEncA() {
  if (g_head_pwm > 0) {
    g_head_last_dir = 1;
    g_head_ticks++;
  } else if (g_head_pwm < 0) {
    g_head_last_dir = -1;
    g_head_ticks--;
  } else {
    // Frenleme/atalet aninda son hareket yonunde sayarak faz terslenmesi ve kaymayi onle
    g_head_ticks += g_head_last_dir;
  }
}






// Erken donanımsal pin kilidi: MCU açıldığı mikrosaniyede (C runtime ve main'den önce)
// tüm motor PWM pinlerini kesin olarak OUTPUT ve LOW yaparak BTS7960 açılış savrulmasını sıfırlar.
void init_early_pwm(void) __attribute__((naked)) __attribute__((section(".init3")));
void init_early_pwm(void) {
  // Pin 44 (PL5) & Pin 45 (PL4) - Kafa Motoru
  PORTL &= ~((1 << 5) | (1 << 4));
  DDRL  |=  ((1 << 5) | (1 << 4));

  // Pin 5 (PE3) & Pin 6 (PH3) - Sol Tekerlek
  PORTE &= ~(1 << 3);
  DDRE  |=  (1 << 3);
  PORTH &= ~(1 << 3);
  DDRH  |=  (1 << 3);

  // Pin 9 (PH6) & Pin 10 (PB4) - Sağ Tekerlek
  PORTH &= ~(1 << 6);
  DDRH  |=  (1 << 6);
  PORTB &= ~(1 << 4);
  DDRB  |=  (1 << 4);
}

void setupIO() {
  // Önce pinleri LOW'a çek, sonra OUTPUT yap (BTS7960 açılış anlık darbe koruması)

  digitalWrite(L_MOTOR_PWM_FWD, LOW);
  digitalWrite(L_MOTOR_PWM_REV, LOW);
  digitalWrite(R_MOTOR_PWM_FWD, LOW);
  digitalWrite(R_MOTOR_PWM_REV, LOW);
  digitalWrite(HEAD_MOTOR_PWM_FWD, LOW);
  digitalWrite(HEAD_MOTOR_PWM_REV, LOW);

  pinMode(STATUS_LED, OUTPUT);
  pinMode(L_MOTOR_PWM_FWD, OUTPUT);
  pinMode(L_MOTOR_PWM_REV, OUTPUT);
  pinMode(R_MOTOR_PWM_FWD, OUTPUT);
  pinMode(R_MOTOR_PWM_REV, OUTPUT);
  pinMode(HEAD_MOTOR_PWM_FWD, OUTPUT);
  pinMode(HEAD_MOTOR_PWM_REV, OUTPUT);


  pinMode(L_ENC_A, INPUT_PULLUP);
  pinMode(L_ENC_B, INPUT_PULLUP);
  pinMode(R_ENC_A, INPUT_PULLUP);
  pinMode(R_ENC_B, INPUT_PULLUP);
  pinMode(HEAD_ENC_A, INPUT_PULLUP);
  pinMode(HEAD_ENC_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(L_ENC_A),    leftEncA,  RISING);
  attachInterrupt(digitalPinToInterrupt(R_ENC_A),    rightEncA, RISING);
  attachInterrupt(digitalPinToInterrupt(HEAD_ENC_A), headEncA,  RISING);

  // PWM frekansı -> 8-bit phase-correct, prescaler 1: 16MHz/(1*510) = 31.37 kHz
  // Timer2: pin 9,10 (sağ)   Timer3: pin 5 (sol ileri)
  // Timer4: pin 6 (sol geri) Timer5: pin 44,45 (kafa)
  TCCR2B = (TCCR2B & 0xF8) | 0x01;
  TCCR3B = (TCCR3B & 0xF8) | 0x01;
  TCCR4B = (TCCR4B & 0xF8) | 0x01;
  TCCR5B = (TCCR5B & 0xF8) | 0x01;
}

void stopMotors() {
  setLeftPWM(0);
  setRightPWM(0);
  setHeadPWM(0);
}

void publishEncoders(uint32_t dt_us, int32_t dl, int32_t dr) {
  int32_t head_ticks = readTicks(g_head_ticks);
  uint8_t payload[4 + 4 + 4 + 4];
  memcpy(&payload[0], &dl, 4);
  memcpy(&payload[4], &dr, 4);
  memcpy(&payload[8], &head_ticks, 4);
  memcpy(&payload[12], &dt_us, 4);
  Proto::writePacket(Serial, Proto::ENCODER_TICKS, payload, sizeof(payload));
}


void publishDiag(uint16_t vbat_mV, int16_t temp_cX100, uint32_t flags) {
  uint8_t payload[2 + 2 + 4];
  memcpy(&payload[0], &vbat_mV, 2);
  memcpy(&payload[2], &temp_cX100, 2);
  memcpy(&payload[4], &flags, 4);
  Proto::writePacket(Serial, Proto::DIAGNOSTICS, payload, sizeof(payload));
}

// Kafa konum PID'i. Limit switch olmadigi icin stall korumasi sart:
// mekanik dayanaga dayanirsa motor akim ceker ve isinir.
void headControl(uint32_t dt_ms) {
  int32_t pos = readTicks(g_head_ticks);
  int32_t err = g_head_target_ticks - pos;

  if (!g_motors_enabled || !g_head_active) {
    setHeadPWM(0);
    g_head_err_prev = err;
    g_head_stall_ref = pos;
    g_head_stall_ms = millis();
    return;
  }


  if (abs(err) <= HEAD_DEADBAND_TICKS) {
    setHeadPWM(0);
    g_head_err_prev = err;
    g_head_stall_ref = pos;
    g_head_stall_ms = millis();
    return;
  }

  float de = (float)(err - g_head_err_prev) / (dt_ms / 1000.0f);
  g_head_err_prev = err;

  // PID + Statik sürtünme eşiği için feedforward tabanı
  float ff = (err > 0) ? (float)HEAD_PWM_MIN : -(float)HEAD_PWM_MIN;
  float u = ff + (HEAD_KP * (float)err) + (HEAD_KD * de);
  int pwm = (int)constrain(u, (float)-HEAD_PWM_LIMIT, (float)HEAD_PWM_LIMIT);

  setHeadPWM(pwm);

  // Stall tespiti: PWM veriliyor ama enkoder kımıldamıyor
  if (abs(pos - g_head_stall_ref) >= 2) {
    g_head_stall_ref = pos;
    g_head_stall_ms = millis();
  } else if (millis() - g_head_stall_ms > HEAD_STALL_MS) {
    setHeadPWM(0);
    // Hedefi bulunduğu yere çek: aksi halde motor dayanağa dayanmaya devam eder
    g_head_target_ticks = pos;
    g_head_err_prev = 0;
    g_diag_flags |= FLAG_HEAD_STALL;
  }
}

void loopControl() {
  uint32_t now = millis();
  if (now - g_last_control_ms < CONTROL_DT_MS) return;
  uint32_t dt_ms = now - g_last_control_ms;
  g_last_control_ms = now;

  // Watchdog: 500 ms içinde heartbeat/komut gelmezse motorları kes
  if (now - g_last_heartbeat_ms > 500) {
    g_motors_enabled = false;
    stopMotors();
    g_diag_flags |= FLAG_WATCHDOG_TIMEOUT;
  } else {
    g_motors_enabled = true;
    g_diag_flags &= ~FLAG_WATCHDOG_TIMEOUT;
  }

  // ✅ FIX: Enkoder okuma atomik yap (AVR'de int32_t atomik değil)
  int32_t l_ticks = readTicks(g_left_ticks);
  int32_t r_ticks = readTicks(g_right_ticks);

  int32_t dl = l_ticks - g_left_last_ticks;
  int32_t dr = r_ticks - g_right_last_ticks;
  g_left_last_ticks = l_ticks;
  g_right_last_ticks = r_ticks;

  float dt_min = dt_ms / 60000.0f; // ms -> dakika
  float l_rpm_meas = (dl / (float)TICKS_PER_REV_L) / dt_min;
  float r_rpm_meas = (dr / (float)TICKS_PER_REV_R) / dt_min;

  // ✅ FIX: PID anti-windup iyileştirildi (conditional integration)
  auto pid_step = [&](float target_rpm, float meas_rpm, float& e_i, float& e_prev)->int {
    if (abs(target_rpm) < 0.01f) {
      e_i = 0.0f;
      e_prev = 0.0f;
      return 0;
    }
    float e = target_rpm - meas_rpm;
    float de = (e - e_prev) / (dt_ms / 1000.0f);
    e_prev = e;
    
    // Feedforward: motor statik sürtünme (stiction) eşiğini aşmak için minimum PWM tabanı
    float ff = (target_rpm > 0.0f) ? 25.0f : -25.0f;
    float u = ff + (KP * 2.0f * e) + (KI * e_i) + (KD * de);
    int pwm = (int)constrain(u, -PWM_MAX, PWM_MAX);
    
    // Conditional integration: sadece PWM saturate olmadığında integral artır
    if (abs(pwm) < PWM_MAX) {
      e_i += e * (dt_ms / 1000.0f);
      e_i = constrain(e_i, -PID_INTEGRAL_LIMIT, PID_INTEGRAL_LIMIT);
    }
    
    return pwm;
  };

  int l_pwm = 0, r_pwm = 0;
  if (g_motors_enabled) {
    l_pwm = pid_step(g_left_target_rpm, l_rpm_meas, g_left_err_i, g_left_err_prev);
    r_pwm = pid_step(g_right_target_rpm, r_rpm_meas, g_right_err_i, g_right_err_prev);
  }
  setLeftPWM(l_pwm);
  setRightPWM(r_pwm);

  // Kafa konum kontrolü (aynı 50 Hz döngüde)
  headControl(dt_ms);

  publishEncoders(dt_ms * 1000u, dl, dr);

  // Basit diagnostik: bayraklar
  uint16_t vbat = 12000; // mV (örn. gelecekte ADC ile ölç)
  int16_t temp = 2500;   // 25.00 C
  publishDiag(vbat, temp, g_diag_flags);

  // 1 saniyelik Serial2 durum telemetrisi (UART0 binary akışına asla dokunmaz)
  if (now - g_last_diag_serial2_ms >= 1000) {
    g_last_diag_serial2_ms = now;
    Serial2.print(F("[MCU STATUS 1s] hb_rx="));
    Serial2.print(g_hb_rx_count);
    Serial2.print(F(" hb_ack="));
    Serial2.print(g_hb_ack_tx_count);
    Serial2.print(F(" mot_en="));
    Serial2.print(g_motors_enabled ? 1 : 0);
    Serial2.print(F(" head_ticks="));
    Serial2.print(readTicks(g_head_ticks));
    Serial2.print(F(" tx_avail="));
    Serial2.println(Serial.availableForWrite());
  }
}

void processPacket(uint8_t msg_id, const uint8_t* pl, uint8_t len) {
  switch (msg_id) {
    case Proto::HEARTBEAT: {
      g_hb_rx_count++;
      g_last_heartbeat_ms = millis();
      digitalWrite(STATUS_LED, !digitalRead(STATUS_LED));

      uint32_t seq = 0;
      if (len >= 4 && pl != nullptr) {
        memcpy(&seq, pl, 4);
      }

      int buf_before = Serial.availableForWrite();
      Serial2.print(F("[HB RX] id=0x01 len="));
      Serial2.print(len);
      Serial2.print(F(" seq="));
      Serial2.print(seq);
      Serial2.print(F(" tx_buf_before="));
      Serial2.println(buf_before);

      Serial2.print(F("[HB ACK TX BEGIN] seq="));
      Serial2.println(seq);

      Proto::writePacket(Serial, Proto::HEARTBEAT_ACK, pl, len);
      g_hb_ack_tx_count++;

      int buf_after = Serial.availableForWrite();
      Serial2.print(F("[HB ACK TX END] count="));
      Serial2.print(g_hb_ack_tx_count);
      Serial2.print(F(" tx_buf_after="));
      Serial2.println(buf_after);
    } break;
    case Proto::WHEEL_CMD: {
      if (len < 8) break;
      memcpy(&g_left_target_rpm, &pl[0], 4);
      memcpy(&g_right_target_rpm, &pl[4], 4);
      g_last_heartbeat_ms = millis(); // komut da heartbeat sayılır
      Serial2.print(F("[WHEEL CMD] L="));
      Serial2.print(g_left_target_rpm);
      Serial2.print(F(" R="));
      Serial2.println(g_right_target_rpm);
    } break;
    case Proto::HEAD_CMD: {
      if (len < 4) break;
      float angle_deg;
      memcpy(&angle_deg, &pl[0], 4);

      // Yazılımsal limit: limit switch yok, mekanik dayanağa gitmeyi engelle
      float clamped = constrain(angle_deg, HEAD_MIN_DEG, HEAD_MAX_DEG);
      if (clamped != angle_deg) g_diag_flags |= FLAG_HEAD_LIMIT;
      else                      g_diag_flags &= ~FLAG_HEAD_LIMIT;

      g_head_target_ticks = (int32_t)lroundf(clamped * HEAD_TICKS_PER_DEG);
      g_head_active = true;
      // Yeni hedef geldi: eski stall kilidini kaldır ve anlık konumu referans al

      g_head_stall_ref = readTicks(g_head_ticks);
      g_head_stall_ms = millis();
      g_diag_flags &= ~FLAG_HEAD_STALL;
      g_last_heartbeat_ms = millis();
      Serial2.print(F("[HEAD CMD] angle="));
      Serial2.println(angle_deg);
    } break;
  }
}

void setup() {
  setupIO();
  stopMotors();
  Serial.begin(SERIAL_BAUD);

  // İkincil debug portu (Mega Pin 16 TX2, Pin 17 RX2)
  Serial2.begin(115200);
  Serial2.println(F("[ASTRO MCU BOOT] Protocol=v2.0 Baud=115200 Serial2=DebugReady"));

  // Açılıştaki kafa konumu 0° kabul edilir (limit switch / homing yok)
  g_head_target_ticks = 0;
  g_head_stall_ms = millis();

  g_last_control_ms = millis();
  g_last_heartbeat_ms = millis();
  g_last_diag_serial2_ms = millis();

  // ✅ FIX: Watchdog timeout 2s'ye çıkarıldı (güvenlik marjı)
  wdt_enable(WDTO_2S);
}

void loop() {
  // Watchdog besle
  wdt_reset();

  // Seri parser
  static Proto::Parser parser;
  while (Serial.available() > 0) {
    uint8_t b = Serial.read();
    uint8_t id; const uint8_t* payload; uint8_t pl_len;
    if (parser.feed(b, id, payload, pl_len)) {
      processPacket(id, payload, pl_len);
    }
  }

  // Kontrol döngüsü 50 Hz
  loopControl();
}
