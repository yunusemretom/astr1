#pragma once

/*
 * ASTRO alt kontrol - pin haritasi (Arduino Mega 2560)
 * ----------------------------------------------------
 * Sahadaki gercek kablolamayi yansitir.
 *
 * DIKKAT - bu pin dagilimi iki seyi ZORUNLU olarak devre disi birakir:
 *
 *   1) MPU-6050 IMU:  20 = SDA, 21 = SCL. Bu pinler kafa enkoderine
 *      verildigi icin donanimsal I2C kullanilamaz. Mega'da I2C baska
 *      pine tasinamaz. Firmware'de IMU kodu kaldirildi; /imu/data_raw
 *      artik yayinlanmiyor (bkz. ekf.yaml -> imu0).
 *
 *   2) TMC2209 step surucu:  Serial1 (18 = TX1, 19 = RX1) sag tekerlek
 *      enkoderine verildi. Kafa zaten step motordan BTS7960'li DC
 *      motora gecti, TMC2209 kodu kaldirildi.
 *
 * Mega'nin 6 dis kesme pininin tamami (2, 3, 18, 19, 20, 21) enkoderlere
 * ayrildi. Yeni bir kesme kaynagi eklemek icin pin-change interrupt
 * gerekir.
 */

// ─────────────────────────────────────────────────────────────
//  Motor suruculeri - BTS7960 (RPWM / LPWM)
//  R_EN ve L_EN uclari 5V'a sabit: suruculer daima aktif, MCU
//  tarafindan donanimsal olarak kesilemez. Durdurma = PWM 0.
// ─────────────────────────────────────────────────────────────
#define L_MOTOR_PWM_FWD    5   // sol  RPWM  -> Timer3A
#define L_MOTOR_PWM_REV    6   // sol  LPWM  -> Timer4A
#define R_MOTOR_PWM_FWD    9   // sag  RPWM  -> Timer2B
#define R_MOTOR_PWM_REV   10   // sag  LPWM  -> Timer2A
#define HEAD_MOTOR_PWM_FWD 45  // kafa Sol/Pozitif PWM (CCW) -> Timer5B
#define HEAD_MOTOR_PWM_REV 44  // kafa Sag/Negatif PWM (CW)  -> Timer5C




// ─────────────────────────────────────────────────────────────
//  Enkoderler - A kanallari kesme pininde, B kanallari yon icin okunur
// ─────────────────────────────────────────────────────────────
#define L_ENC_A     2   // INT0
#define L_ENC_B     3   // INT1 (kesme baglanmiyor, sadece yon okunuyor)
#define R_ENC_A    18   // INT5  (eski Serial1 TX1)
#define R_ENC_B    19   // INT4  (eski Serial1 RX1)
#define HEAD_ENC_A 20   // INT3  (eski I2C SDA)
#define HEAD_ENC_B 21   // INT2  (eski I2C SCL)

// ─────────────────────────────────────────────────────────────
//  Durum LED'i
// ─────────────────────────────────────────────────────────────
#define STATUS_LED 13
