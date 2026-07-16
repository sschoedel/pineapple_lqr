"""Xsens MTi IMU driver (XBus/MTData2 over USB serial) for the robot runtime.

The V3's Xsens MTi connects over USB (shows up as /dev/serial/by-id/usb-Xsens*
or /dev/ttyUSB*). This speaks the XBus protocol directly with pyserial — no
Xsens SDK needed on the Pi. The CSL bridge (pineapple_hardware_interface)
used the same device via the XDA SDK with Quaternion + RateOfTurn +
Acceleration outputs; we request the same set at 100/200/200 Hz.

The balance controller needs, per tick (base frame: x fwd, y left, z up):
  quat  (4,) w,x,y,z body orientation (yaw reference irrelevant — the
        controller is heading-invariant; roll/pitch must be gravity-true)
  gyro  (3,) rad/s body angular rates
  accel (3,) m/s^2 raw accelerometer (includes gravity reaction)

MOUNT ORIENTATION: set MOUNT_QUAT below to the rotation from the sensor
frame to the robot base frame (identity if the MTi's x-axis points at the
robot's front with the label up). VERIFY before first balance: tilt the
robot nose-down -> pitch must go NEGATIVE... actually verify with the GUI
readout: nose-down = pitch < 0? The controller convention (MJCF base) is
pitch positive = nose down about +y(left). Tilt and check signs.

Parser unit tests: motor_control tests aren't set up; the parser is tested
in pineapple_lqr/tests/test_xsens_parser.py against synthetic frames.
"""

from __future__ import annotations

import glob
import struct
import threading
import time

import numpy as np

try:
    import serial
except ImportError:  # pyserial is in pyproject; be import-safe anyway
    serial = None

PREAMBLE = 0xFA
BID = 0xFF
MID_GOTO_CONFIG = 0x30
MID_GOTO_CONFIG_ACK = 0x31
MID_SET_OUTPUT_CFG = 0xC0
MID_GOTO_MEASUREMENT = 0x10
MID_MTDATA2 = 0x36

XDI_PACKET_COUNTER = 0x1020
XDI_SAMPLE_TIME_FINE = 0x1060
XDI_QUATERNION = 0x2010
XDI_ACCELERATION = 0x4020
XDI_RATE_OF_TURN = 0x8020

STALE_S = 0.05

# Rotation sensor frame -> robot base frame (w, x, y, z). Identity assumes
# the MTi x-axis points forward, z up. Adjust after the tilt-sign check.
MOUNT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rotate(q, v):
    w = q[0]
    u = q[1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def xbus_frame(mid: int, payload: bytes = b"") -> bytes:
    """Build an XBus frame (standard length only; payloads < 255)."""
    body = bytes([BID, mid, len(payload)]) + payload
    cksum = (-sum(body)) & 0xFF
    return bytes([PREAMBLE]) + body + bytes([cksum])


class XbusParser:
    """Incremental XBus frame parser: feed() bytes, yields (mid, payload)."""

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes):
        self._buf.extend(data)
        out = []
        while True:
            i = self._buf.find(bytes([PREAMBLE]))
            if i < 0:
                self._buf.clear()
                break
            if i > 0:
                del self._buf[:i]
            if len(self._buf) < 5:
                break
            if self._buf[1] != BID:
                del self._buf[0]  # false preamble inside noise — rescan
                continue
            length = self._buf[3]
            if length == 0xFF:  # extended-length frame
                if len(self._buf) < 6:
                    break
                ext = (self._buf[4] << 8) | self._buf[5]
                total = 6 + ext + 1
                payload_at = 6
                plen = ext
            else:
                total = 4 + length + 1
                payload_at = 4
                plen = length
            if len(self._buf) < total:
                break
            frame = bytes(self._buf[:total])
            if sum(frame[1:]) & 0xFF != 0:
                del self._buf[0]  # corrupt frame — advance one byte, rescan
                continue
            del self._buf[:total]
            out.append((frame[2], frame[payload_at:payload_at + plen]))
        return out


def parse_mtdata2(payload: bytes) -> dict:
    """Extract quaternion / gyro / accel (float32 BE) from an MTData2 body."""
    out = {}
    i = 0
    n = len(payload)
    while i + 3 <= n:
        data_id = (payload[i] << 8) | payload[i + 1]
        dlen = payload[i + 2]
        chunk = payload[i + 3:i + 3 + dlen]
        i += 3 + dlen
        base_id = data_id & 0xFFF0  # low nibble = precision/format flags
        try:
            if base_id == XDI_QUATERNION and dlen >= 16:
                out["quat"] = np.array(struct.unpack(">4f", chunk[:16]))
            elif base_id == XDI_RATE_OF_TURN and dlen >= 12:
                out["gyro"] = np.array(struct.unpack(">3f", chunk[:12]))
            elif base_id == XDI_ACCELERATION and dlen >= 12:
                out["accel"] = np.array(struct.unpack(">3f", chunk[:12]))
        except struct.error:
            pass
    return out


def find_port() -> str | None:
    for pattern in ("/dev/serial/by-id/*Xsens*", "/dev/serial/by-id/*xsens*",
                    "/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


class XsensMtiImu:
    """Threaded MTi reader. read() -> (quat, gyro, accel) or None if stale."""

    available = True

    def __init__(self, port: str | None = None, baud: int = 115200,
                 configure: bool = True):
        if serial is None:
            raise RuntimeError("pyserial not installed")
        port = port or find_port()
        if port is None:
            raise RuntimeError("no Xsens/USB-serial device found")
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.port = port
        self._parser = XbusParser()
        self._lock = threading.Lock()
        self._quat = None
        self._gyro = None
        self._accel = None
        self._t = 0.0
        self._mount = MOUNT_QUAT / np.linalg.norm(MOUNT_QUAT)
        if configure:
            self._configure()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _configure(self):
        """Best-effort output configuration (quat@100, gyro/accel@200)."""
        def send(mid, payload=b""):
            self.ser.write(xbus_frame(mid, payload))
            time.sleep(0.05)

        send(MID_GOTO_CONFIG)
        time.sleep(0.2)
        self.ser.reset_input_buffer()
        cfg = b"".join(
            struct.pack(">HH", xdi, rate)
            for xdi, rate in (
                (XDI_PACKET_COUNTER, 0xFFFF),
                (XDI_SAMPLE_TIME_FINE, 0xFFFF),
                (XDI_QUATERNION, 100),
                (XDI_RATE_OF_TURN, 200),
                (XDI_ACCELERATION, 200),
            )
        )
        send(MID_SET_OUTPUT_CFG, cfg)
        send(MID_GOTO_MEASUREMENT)
        self.ser.reset_input_buffer()

    def _reader(self):
        while self._running:
            try:
                data = self.ser.read(512)
            except (OSError, serial.SerialException):
                time.sleep(0.1)
                continue
            if not data:
                continue
            for mid, payload in self._parser.feed(data):
                if mid != MID_MTDATA2:
                    continue
                fields = parse_mtdata2(payload)
                with self._lock:
                    if "quat" in fields:
                        self._quat = fields["quat"]
                    if "gyro" in fields:
                        self._gyro = fields["gyro"]
                    if "accel" in fields:
                        self._accel = fields["accel"]
                    if fields:
                        self._t = time.monotonic()

    def read(self):
        with self._lock:
            if self._quat is None or self._gyro is None or self._accel is None:
                return None
            if time.monotonic() - self._t > STALE_S:
                return None
            quat = self._quat.copy()
            gyro = self._gyro.copy()
            accel = self._accel.copy()
        # sensor frame -> base frame via mount rotation
        m = self._mount
        quat = _quat_mul(quat, _quat_conj(m))
        gyro = _rotate(m, gyro)
        accel = _rotate(m, accel)
        n = np.linalg.norm(quat)
        if n < 1e-6:
            return None
        return quat / n, gyro, accel

    def close(self):
        self._running = False
        self._thread.join(timeout=0.5)
        try:
            self.ser.close()
        except Exception:
            pass


class NoImu:
    available = False

    def read(self):
        return None


def create():
    """Return the robot's IMU driver, or NoImu() if none is present."""
    try:
        imu = XsensMtiImu()
        print(f"Xsens MTi on {imu.port}")
        return imu
    except Exception as e:
        print(f"IMU unavailable: {e}")
        return NoImu()
