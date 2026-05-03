"""MPU-9250 IMU reader for Raspberry Pi 4 Model B via I2C (smbus2).

Wiring (Pi 5 GPIO header):
    SDA  -> GPIO 2  (pin 3)
    SCL  -> GPIO 3  (pin 5)
    VCC  -> 3.3 V   (pin 1)
    GND  -> GND     (pin 6)
    AD0  -> GND     -> I2C address 0x68  (or 3.3 V for 0x69)

The AK8963 magnetometer is accessed via the MPU-9250 I2C bypass mode.

Requires:
    pip install smbus2

Scale factors used:
    Accel  ±2 g    : 16384 LSB/g  -> m/s²
    Gyro   ±250 dps: 131 LSB/°/s  -> rad/s
    Mag    16-bit  : 0.15 µT/LSB
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

from src.core.types import IMUReading

logger = logging.getLogger(__name__)

try:
    import smbus2  # type: ignore
    _SMBUS2_OK = True
except ImportError:
    _SMBUS2_OK = False

# ── MPU-9250 register map ─────────────────────────────────────────────────────
_MPU_ADDR_DEFAULT = 0x68
_PWR_MGMT_1  = 0x6B
_PWR_MGMT_2  = 0x6C
_GYRO_CONFIG  = 0x1B   # 0x00 = ±250 dps
_ACCEL_CONFIG = 0x1C   # 0x00 = ±2 g
_INT_PIN_CFG  = 0x37   # bit 1 = I2C bypass enable for AK8963
_ACCEL_XOUT_H = 0x3B   # 6 bytes: ax_H, ax_L, ay_H, ay_L, az_H, az_L
_GYRO_XOUT_H  = 0x43   # 6 bytes: gx_H, gx_L, gy_H, gy_L, gz_H, gz_L

# ── AK8963 magnetometer ───────────────────────────────────────────────────────
_AK_ADDR   = 0x0C
_AK_CNTL   = 0x0A   # write 0x16: continuous mode 2 (100 Hz, 16-bit)
_AK_HXL    = 0x03   # 7 bytes: HXL, HXH, HYL, HYH, HZL, HZH, ST2

# ── scale constants ───────────────────────────────────────────────────────────
_ACCEL_SCALE = 9.80665 / 16384.0          # m/s² per LSB  (±2 g)
_GYRO_SCALE  = (math.pi / 180.0) / 131.0  # rad/s per LSB (±250 dps)
_MAG_SCALE   = 0.15                        # µT per LSB    (AK8963 16-bit)


def _s16(high: int, low: int) -> int:
    """Combine two bytes into a signed 16-bit integer."""
    val = (high << 8) | low
    return val - 65536 if val >= 32768 else val


class MPU9250Reader:
    """Read accel, gyro and magnetometer from an MPU-9250 over I2C.

    Parameters
    ----------
    bus :
        I2C bus number (1 on Raspberry Pi 4 Model B).
    address :
        MPU-9250 I2C address (0x68 when AD0=GND, 0x69 when AD0=3.3V).
    """

    def __init__(self, bus: int = 1, address: int = _MPU_ADDR_DEFAULT) -> None:
        if not _SMBUS2_OK:
            raise RuntimeError(
                "smbus2 is not installed. Run: pip install smbus2"
            )
        self._bus = smbus2.SMBus(bus)
        self._addr = address
        self._init_device()
        logger.info("MPU-9250 initialised on I2C bus %d addr 0x%02X", bus, address)

    # ── device initialisation ─────────────────────────────────────────────────

    def _init_device(self) -> None:
        b = self._bus
        a = self._addr

        b.write_byte_data(a, _PWR_MGMT_1, 0x00)   # wake up
        time.sleep(0.1)
        b.write_byte_data(a, _PWR_MGMT_2, 0x00)   # enable accel + gyro
        b.write_byte_data(a, _ACCEL_CONFIG, 0x00)  # ±2 g
        b.write_byte_data(a, _GYRO_CONFIG, 0x00)   # ±250 dps
        b.write_byte_data(a, _INT_PIN_CFG, 0x02)   # enable I2C bypass
        time.sleep(0.01)
        b.write_byte_data(_AK_ADDR, _AK_CNTL, 0x16)  # AK8963 cont. mode 2
        time.sleep(0.01)

    # ── public interface (matches IMUSimulator) ────────────────────────────────

    def read(self, timestamp: float) -> IMUReading:
        b = self._bus
        a = self._addr

        raw_a = b.read_i2c_block_data(a, _ACCEL_XOUT_H, 6)
        ax = _s16(raw_a[0], raw_a[1]) * _ACCEL_SCALE
        ay = _s16(raw_a[2], raw_a[3]) * _ACCEL_SCALE
        az = _s16(raw_a[4], raw_a[5]) * _ACCEL_SCALE

        raw_g = b.read_i2c_block_data(a, _GYRO_XOUT_H, 6)
        gx = _s16(raw_g[0], raw_g[1]) * _GYRO_SCALE
        gy = _s16(raw_g[2], raw_g[3]) * _GYRO_SCALE
        gz = _s16(raw_g[4], raw_g[5]) * _GYRO_SCALE

        try:
            raw_m = b.read_i2c_block_data(_AK_ADDR, _AK_HXL, 7)
            mx = _s16(raw_m[1], raw_m[0]) * _MAG_SCALE
            my = _s16(raw_m[3], raw_m[2]) * _MAG_SCALE
            mz = _s16(raw_m[5], raw_m[4]) * _MAG_SCALE
        except Exception:
            mx, my, mz = 0.0, 0.0, 0.0

        return IMUReading(
            accel=(ax, ay, az),
            gyro=(gx, gy, gz),
            mag=(mx, my, mz),
            timestamp=timestamp,
        )

    @property
    def source_name(self) -> str:
        return "hardware"

    def cycle_scenario(self) -> None:
        pass

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:
            pass
