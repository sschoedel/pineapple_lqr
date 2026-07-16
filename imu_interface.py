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


def _decode_floats(chunk: bytes, count: int, precision: int):
    """Decode `count` values per the data-id precision flag (bits 0-1):
    0 = float32, 1 = fixed 12.20, 2 = fixed 16.32, 3 = float64."""
    try:
        if precision == 0:
            need = 4 * count
            return np.array(struct.unpack(f">{count}f", chunk[:need])) \
                if len(chunk) >= need else None
        if precision == 3:
            need = 8 * count
            return np.array(struct.unpack(f">{count}d", chunk[:need])) \
                if len(chunk) >= need else None
        if precision == 1:  # FP12.20: int32 / 2^20
            need = 4 * count
            if len(chunk) < need:
                return None
            raw = struct.unpack(f">{count}i", chunk[:need])
            return np.array(raw) / float(1 << 20)
        if precision == 2:  # FP16.32: 32-bit fraction + 16-bit integer
            need = 6 * count
            if len(chunk) < need:
                return None
            vals = []
            for k in range(count):
                frac, integ = struct.unpack(
                    ">Ih", chunk[6 * k:6 * k + 6])
                vals.append(integ + frac / float(1 << 32))
            return np.array(vals)
    except struct.error:
        return None
    return None


def parse_mtdata2(payload: bytes) -> dict:
    """Extract quaternion / gyro / accel from an MTData2 body, honouring
    the precision flag (bits 0-1) and reporting the coordinate-system flag
    (bits 2-3: 0=ENU, 1=NED, 2=NWU) of the quaternion."""
    out = {}
    i = 0
    n = len(payload)
    while i + 3 <= n:
        data_id = (payload[i] << 8) | payload[i + 1]
        dlen = payload[i + 2]
        chunk = payload[i + 3:i + 3 + dlen]
        i += 3 + dlen
        base_id = data_id & 0xFFF0
        precision = data_id & 0x3
        if base_id == XDI_QUATERNION:
            v = _decode_floats(chunk, 4, precision)
            if v is not None:
                out["quat"] = v
                out["coord"] = (data_id >> 2) & 0x3
        elif base_id == XDI_RATE_OF_TURN:
            v = _decode_floats(chunk, 3, precision)
            if v is not None:
                out["gyro"] = v
        elif base_id == XDI_ACCELERATION:
            v = _decode_floats(chunk, 3, precision)
            if v is not None:
                out["accel"] = v
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
        self.ser = serial.Serial(port, baud, timeout=0.02)
        self.port = port
        # FTDI adapters buffer the stream into large bursts by default
        # (measured ~78 ms clumps on the Pi with the Xsens USB converter,
        # which quantized the orientation into visible jumps). Low-latency
        # mode sets the chip flush timer to 1 ms. A udev rule on the Pi
        # (99-xsens-latency.rules) does the same at plug time; this is the
        # in-process fallback and needs permissions (the GUI runs as root).
        try:
            self.ser.set_low_latency_mode(True)
        except Exception as e:
            print(f"xsens: could not set low-latency mode ({e}) — "
                  "install the udev rule (see README)")
        self._parser = XbusParser()
        self._lock = threading.Lock()
        self._quat = None
        self._gyro = None
        self._accel = None
        self._t = 0.0
        self._mount = MOUNT_QUAT / np.linalg.norm(MOUNT_QUAT)
        self.stats = {"n_quat": 0, "n_gyro": 0, "n_accel": 0,
                      "bad_quat": 0, "coord": -1, "t0": time.monotonic()}
        if configure:
            self._configure()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _configure(self):
        """Output configuration (quat@100, gyro/accel@200) with ack checks.
        Failure is non-fatal: the device keeps its stored configuration and
        we report what happened (a pre-configured 100 Hz stream is fine)."""

        def send_wait_ack(mid, payload=b"", ack_mid=None, wait=0.4):
            self.ser.write(xbus_frame(mid, payload))
            parser = XbusParser()
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                data = self.ser.read(self.ser.in_waiting or 1)
                for got_mid, _pl in parser.feed(data):
                    if ack_mid is not None and got_mid == ack_mid:
                        return True
            return ack_mid is None

        ok_cfg = send_wait_ack(MID_GOTO_CONFIG, ack_mid=MID_GOTO_CONFIG_ACK)
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
        ok_out = send_wait_ack(MID_SET_OUTPUT_CFG, cfg,
                               ack_mid=MID_SET_OUTPUT_CFG + 1)
        send_wait_ack(MID_GOTO_MEASUREMENT, ack_mid=MID_GOTO_MEASUREMENT + 1)
        self.configured = bool(ok_cfg and ok_out)
        if not self.configured:
            print("xsens: output config not acknowledged — using the "
                  "device's stored configuration")
        self.ser.reset_input_buffer()

    # NED -> z-up: rotate the world side 180 deg about x (N,E,D)->(N,W,U).
    _Q_X180 = np.array([0.0, 1.0, 0.0, 0.0])

    def _reader(self):
        while self._running:
            try:
                # block until at least one byte (short timeout), then drain
                # everything pending — per-packet latency ~1 ms instead of
                # the read(512)-or-timeout batching that quantized samples
                # into ~50 ms steps
                data = self.ser.read(1)
                if data and self.ser.in_waiting:
                    data += self.ser.read(self.ser.in_waiting)
            except (OSError, serial.SerialException):
                time.sleep(0.12)  # not 0.1: the fastsleep shim clamps exact-100ms sleeps
                continue
            if not data:
                continue
            for mid, payload in self._parser.feed(data):
                if mid != MID_MTDATA2:
                    continue
                fields = parse_mtdata2(payload)
                quat = fields.get("quat")
                if quat is not None:
                    self.stats["coord"] = fields.get("coord", 0)
                    if abs(np.linalg.norm(quat) - 1.0) > 0.02:
                        self.stats["bad_quat"] += 1
                        quat = None
                    elif fields.get("coord") == 1:  # NED
                        quat = _quat_mul(self._Q_X180, quat)
                with self._lock:
                    if quat is not None:
                        self._quat = quat
                        self.stats["n_quat"] += 1
                    if "gyro" in fields:
                        self._gyro = fields["gyro"]
                        self.stats["n_gyro"] += 1
                    if "accel" in fields:
                        self._accel = fields["accel"]
                        self.stats["n_accel"] += 1
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
