#pragma once
#include <Arduino.h>

// Paket Yapısı:
// [0xAA][0x55][LEN][MSG_ID][PAYLOAD...][CRC8]
// CRC8 polinom: 0x07, init 0x00 (CRC-8-ATM)
namespace Proto {
  static const uint8_t SOF1 = 0xAA;
  static const uint8_t SOF2 = 0x55;

  enum MsgId : uint8_t {
    HEARTBEAT        = 0x01,  // host->mcu
    WHEEL_CMD        = 0x02,  // host->mcu: float32 left_rpm, right_rpm
    HEAD_CMD         = 0x03,  // host->mcu: float32 angle_deg
    IMU_DATA         = 0x10,  // mcu->host: 6x float32 (ax,ay,az,gx,gy,gz), uint32 micros
    ENCODER_TICKS    = 0x11,  // mcu->host: int32 L, int32 R, uint32 dt_us
    DIAGNOSTICS      = 0x12,  // mcu->host: uint16 vbat_mV, int16 mcu_temp_cX100, uint32 flags
    HEARTBEAT_ACK    = 0x13   // mcu->host
  };

  inline uint8_t crc8(const uint8_t* data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; ++i) {
      crc ^= data[i];
      for (uint8_t b = 0; b < 8; ++b) {
        if (crc & 0x80) crc = (crc << 1) ^ 0x07;
        else crc <<= 1;
      }
    }
    return crc;
  }

  inline void writePacket(Stream& s, uint8_t msg_id, const uint8_t* payload, uint8_t payload_len) {
    uint8_t len = 1 + payload_len; // msg_id + payload
    s.write(SOF1);
    s.write(SOF2);
    s.write(len);
    s.write(msg_id);
    if (payload_len > 0 && payload != nullptr) {
      s.write(payload, payload_len);
    }

    uint8_t buf[1 + 255]; // length + data
    buf[0] = len;
    buf[1] = msg_id;
    if (payload_len > 0 && payload != nullptr) {
      memcpy(&buf[2], payload, payload_len);
    }
    uint8_t crc = crc8(buf, 2 + payload_len);
    s.write(crc);
  }

  struct Parser {
    enum State { WAIT_SOF1, WAIT_SOF2, WAIT_LEN, WAIT_DATA, WAIT_CRC } state = WAIT_SOF1;
    uint8_t len = 0;
    uint8_t data[256];
    uint8_t idx = 0;

    bool feed(uint8_t b, uint8_t& out_id, const uint8_t*& out_payload, uint8_t& out_pl_len) {
      switch (state) {
        case WAIT_SOF1:
          if (b == SOF1) state = WAIT_SOF2;
          break;
        case WAIT_SOF2:
          if (b == SOF2) state = WAIT_LEN;
          else if (b == SOF1) state = WAIT_SOF2;
          else state = WAIT_SOF1;
          break;
        case WAIT_LEN:
          len = b;
          idx = 0;
          state = WAIT_DATA;
          break;
        case WAIT_DATA:
          data[idx++] = b;
          if (idx >= len) state = WAIT_CRC;
          break;
        case WAIT_CRC: {
          uint8_t tmp[1 + 255];
          tmp[0] = len;
          memcpy(&tmp[1], data, len);
          uint8_t calc = crc8(tmp, 1 + len);
          if (calc == b) {
            out_id = data[0];
            out_payload = &data[1];
            out_pl_len = len - 1;
            state = WAIT_SOF1;
            return true;
          } else {
            // CRC fail -> re-sync if current byte is SOF1
            state = (b == SOF1) ? WAIT_SOF2 : WAIT_SOF1;
          }
        } break;
      }
      return false;
    }
  };
}
