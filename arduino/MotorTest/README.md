// ============================================================
// ASTRO V1 - PRODUCTION ARDUINO FIRMWARE (Mega 2560)
// BTS7960 Motor Sürücülü Kafa & Tekerlek Kontrolü
// Baud Rate: 500000 | Non-Blocking ROS2 & ASCII Dual Driver
// ============================================================

#include <Arduino.h>

// =======================
// BTS7960 MOTOR PINLERI
// =======================
#define L_MOTOR_PWM_FWD    5   // Sol Teker İleri (RPWM)
#define L_MOTOR_PWM_REV    6   // Sol Teker Geri  (LPWM)

#define R_MOTOR_PWM_FWD    9   // Sağ Teker İleri (RPWM)
#define R_MOTOR_PWM_REV   10   // Sağ Teker Geri  (LPWM)

#define HEAD_MOTOR_PWM_FWD 44  // Kafa Sağa (RPWM)
#define HEAD_MOTOR_PWM_REV 45  // Kafa Sola (LPWM)

#define STATUS_LED         13

// =======================
// ENKODER PINLERI (Opsiyonel)
// =======================
#define L_ENC_A     2
#define L_ENC_B     3
#define R_ENC_A    18
#define R_ENC_B    19
#define HEAD_ENC_A 20
#define HEAD_ENC_B 21

// =======================
// AYARLAR VE HIZ TAVANLARI
// =======================
static const uint32_t SERIAL_BAUD = 500000;
static const int      HEAD_PWM    = 70;    // Kafa dönüş PWM gücü (yumuşak ve sessiz)
static const int      WHEEL_PWM   = 120;   // Tekerlek PWM gücü
static const uint32_t WATCHDOG_TIMEOUT_MS = 1500; // 1.5 sn komut gelmezse dur

// =======================
// PROTOKOL SABITLERI
// =======================
static const uint8_t SOF1 = 0xAA;
static const uint8_t SOF2 = 0x55;

enum MsgId : uint8_t {
  HEARTBEAT       = 0x01,
  WHEEL_CMD       = 0x02,
  HEAD_CMD        = 0x03,
  HEARTBEAT_ACK   = 0x81,
  ENCODER_TICKS   = 0x82
};

// =======================
// GLOBAL DEGISKENLER
// =======================
volatile int32_t g_head_ticks = 0;
volatile int32_t g_left_ticks = 0;
volatile int32_t g_right_ticks = 0;

static uint32_t g_last_cmd_ms = 0;
static bool     g_head_active = false;

// =======================
// ENKODER KESMELERI
// =======================
void headEncISR()  { g_head_ticks  += digitalRead(HEAD_ENC_B) ? -1 : +1; }
void leftEncISR()  { g_left_ticks  += digitalRead(L_ENC_B)    ? -1 : +1; }
void rightEncISR() { g_right_ticks += digitalRead(R_ENC_B)    ? -1 : +1; }

// =======================
// MOTOR SURUS FONKSIYONLARI
// =======================
void setMotorPWM(int fwd_pin, int rev_pin, int val) {
  val = constrain(val, -255, 255);
  if (val > 0) {
    analogWrite(rev_pin, 0);
    analogWrite(fwd_pin, val);
  } else if (val < 0) {
    analogWrite(fwd_pin, 0);
    analogWrite(rev_pin, -val);
  } else {
    analogWrite(fwd_pin, 0);
    analogWrite(rev_pin, 0);
  }
}

void setHeadPWM(int pwm) {
  setMotorPWM(HEAD_MOTOR_PWM_FWD, HEAD_MOTOR_PWM_REV, pwm);
  g_head_active = (pwm != 0);
  g_last_cmd_ms = millis();
}

void setWheelsPWM(int left_pwm, int right_pwm) {
  setMotorPWM(L_MOTOR_PWM_FWD, L_MOTOR_PWM_REV, left_pwm);
  setMotorPWM(R_MOTOR_PWM_FWD, R_MOTOR_PWM_REV, right_pwm);
  g_last_cmd_ms = millis();
}

void stopAllMotors() {
  analogWrite(L_MOTOR_PWM_FWD, 0);
  analogWrite(L_MOTOR_PWM_REV, 0);
  analogWrite(R_MOTOR_PWM_FWD, 0);
  analogWrite(R_MOTOR_PWM_REV, 0);
  analogWrite(HEAD_MOTOR_PWM_FWD, 0);
  analogWrite(HEAD_MOTOR_PWM_REV, 0);
  g_head_active = false;
}

// =======================
// CRC8 HESABI
// =======================
uint8_t crc8(const uint8_t* data, size_t len) {
  uint8_t crc = 0x00;
  for (size_t i = 0; i < len; ++i) {
    crc ^= data[i];
    for (uint8_t j = 0; j < 8; ++j) {
      if (crc & 0x80) crc = (crc << 1) ^ 0x07;
      else            crc <<= 1;
    }
  }
  return crc;
}

void sendPacket(uint8_t msg_id, const uint8_t* payload, uint8_t len) {
  uint8_t length = 1 + len;
  uint8_t header[4] = { SOF1, SOF2, length, msg_id };
  Serial.write(header, 4);
  if (len > 0 && payload != nullptr) {
    Serial.write(payload, len);
  }
  uint8_t body[256];
  body[0] = length;
  body[1] = msg_id;
  if (len > 0 && payload != nullptr) {
    memcpy(&body[2], payload, len);
  }
  uint8_t c = crc8(body, 2 + len);
  Serial.write(c);
}

// =======================
// BINARY PAKET ISLEYICI
// =======================
void handleBinaryPacket(uint8_t msg_id, const uint8_t* pl, uint8_t len) {
  g_last_cmd_ms = millis();
  
  switch (msg_id) {
    case HEARTBEAT: {
      sendPacket(HEARTBEAT_ACK, pl, len);
      digitalWrite(STATUS_LED, !digitalRead(STATUS_LED));
    } break;

    case HEAD_CMD: {
      if (len >= 4) {
        float angle_deg;
        memcpy(&angle_deg, pl, 4);
        if (angle_deg > 5.0f) {
          setHeadPWM(HEAD_PWM);     // Sağa dön
        } else if (angle_deg < -5.0f) {
          setHeadPWM(-HEAD_PWM);    // Sola dön
        } else {
          setHeadPWM(0);            // Hedefte dur
        }
      }
    } break;

    case WHEEL_CMD: {
      if (len >= 8) {
        float left_rpm, right_rpm;
        memcpy(&left_rpm, &pl[0], 4);
        memcpy(&right_rpm, &pl[4], 4);
        int l_pwm = constrain((int)(left_rpm * 2.5f), -255, 255);
        int r_pwm = constrain((int)(right_rpm * 2.5f), -255, 255);
        setWheelsPWM(l_pwm, r_pwm);
      }
    } break;
  }
}

// =======================
// ASCII KOMUT ISLEYICI
// =======================
void handleAsciiCommand(char c) {
  g_last_cmd_ms = millis();

  switch (c) {
    // Kafa Kontrolleri
    case '5': // Kafa Sağa Dön
      setHeadPWM(HEAD_PWM);
      break;

    case '6': // Kafa Sola Dön
      setHeadPWM(-HEAD_PWM);
      break;

    case '0': // Tüm Motorları Durdur
      stopAllMotors();
      break;

    // Tekerlek Testleri
    case '1': case 'w': case 'W': // İleri
      setWheelsPWM(WHEEL_PWM, WHEEL_PWM);
      break;

    case '2': case 's': case 'S': // Geri
      setWheelsPWM(-WHEEL_PWM, -WHEEL_PWM);
      break;

    case '3': case 'd': case 'D': // Sağa Dön
      setWheelsPWM(WHEEL_PWM, -WHEEL_PWM);
      break;

    case '4': case 'a': case 'A': // Sola Dön
      setWheelsPWM(-WHEEL_PWM, WHEEL_PWM);
      break;

    default:
      break;
  }
}

// =======================
// SERI PROTOKOL AYRISTIRICI (State Machine)
// =======================
void parseSerialStream() {
  static uint8_t state = 0;
  static uint8_t expected_len = 0;
  static uint8_t msg_id = 0;
  static uint8_t payload[64];
  static uint8_t rx_count = 0;

  while (Serial.available() > 0) {
    uint8_t b = Serial.read();

    // 1. Binary Paket Yakalama
    if (state == 0) {
      if (b == SOF1) {
        state = 1;
      } else {
        // Doğrudan ASCII Karakter
        if (b != '\r' && b != '\n') {
          handleAsciiCommand((char)b);
        }
      }
    } else if (state == 1) {
      if (b == SOF2) {
        state = 2;
      } else if (b == SOF1) {
        state = 1;
      } else {
        handleAsciiCommand((char)b);
        state = 0;
      }
    } else if (state == 2) {
      expected_len = b;
      rx_count = 0;
      state = 3;
    } else if (state == 3) {
      if (rx_count == 0) {
        msg_id = b;
      } else if (rx_count - 1 < sizeof(payload)) {
        payload[rx_count - 1] = b;
      }
      rx_count++;
      if (rx_count >= expected_len) {
        state = 4;
      }
    } else if (state == 4) {
      uint8_t body[64];
      body[0] = expected_len;
      body[1] = msg_id;
      if (expected_len > 1) {
        memcpy(&body[2], payload, expected_len - 1);
      }
      uint8_t expected_crc = crc8(body, expected_len + 1);
      if (b == expected_crc) {
        handleBinaryPacket(msg_id, payload, expected_len - 1);
      }
      state = 0;
    }
  }
}

// =======================
// SETUP
// =======================
void setup() {
  Serial.begin(SERIAL_BAUD);

  pinMode(L_MOTOR_PWM_FWD, OUTPUT);
  pinMode(L_MOTOR_PWM_REV, OUTPUT);
  pinMode(R_MOTOR_PWM_FWD, OUTPUT);
  pinMode(R_MOTOR_PWM_REV, OUTPUT);
  pinMode(HEAD_MOTOR_PWM_FWD, OUTPUT);
  pinMode(HEAD_MOTOR_PWM_REV, OUTPUT);
  pinMode(STATUS_LED, OUTPUT);

  // Enkoder pinleri
  pinMode(HEAD_ENC_A, INPUT_PULLUP);
  pinMode(HEAD_ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(HEAD_ENC_A), headEncISR, RISING);

  stopAllMotors();
  g_last_cmd_ms = millis();
}

// =======================
// MAIN LOOP
// =======================
void loop() {
  parseSerialStream();

  // Güvenlik: 1.5 saniye boyunca komut gelmezse motorları durdur
  if (millis() - g_last_cmd_ms > WATCHDOG_TIMEOUT_MS) {
    if (g_head_active) {
      stopAllMotors();
    }
  }
}