"""Unit tests for the Xsens XBus/MTData2 parser (imu_interface.py)."""

import struct

import numpy as np
import pytest

import imu_interface as xi


def mtdata2_payload(quat=None, gyro=None, accel=None):
    body = b""
    if quat is not None:
        body += struct.pack(">HB4f", xi.XDI_QUATERNION, 16, *quat)
    if gyro is not None:
        body += struct.pack(">HB3f", xi.XDI_RATE_OF_TURN, 12, *gyro)
    if accel is not None:
        body += struct.pack(">HB3f", xi.XDI_ACCELERATION, 12, *accel)
    return body


def test_frame_roundtrip():
    payload = mtdata2_payload(quat=(1, 0, 0, 0), gyro=(0.1, -0.2, 0.3),
                              accel=(0, 0, 9.81))
    frame = xi.xbus_frame(xi.MID_MTDATA2, payload)
    p = xi.XbusParser()
    frames = p.feed(frame)
    assert len(frames) == 1
    mid, pl = frames[0]
    assert mid == xi.MID_MTDATA2
    fields = xi.parse_mtdata2(pl)
    assert np.allclose(fields["quat"], [1, 0, 0, 0])
    assert np.allclose(fields["gyro"], [0.1, -0.2, 0.3], atol=1e-6)
    assert np.allclose(fields["accel"], [0, 0, 9.81], atol=1e-5)


def test_fragmented_and_concatenated_feeds():
    f1 = xi.xbus_frame(xi.MID_MTDATA2, mtdata2_payload(gyro=(1, 2, 3)))
    f2 = xi.xbus_frame(xi.MID_MTDATA2, mtdata2_payload(accel=(4, 5, 6)))
    stream = b"\x00\x12" + f1 + b"\xfa\x00garbage" + f2  # noise between frames
    p = xi.XbusParser()
    got = []
    for k in range(0, len(stream), 3):  # drip-feed 3 bytes at a time
        got.extend(p.feed(stream[k:k + 3]))
    assert len(got) == 2
    a = xi.parse_mtdata2(got[0][1])
    b = xi.parse_mtdata2(got[1][1])
    assert np.allclose(a["gyro"], [1, 2, 3], atol=1e-6)
    assert np.allclose(b["accel"], [4, 5, 6], atol=1e-6)


def test_bad_checksum_resync():
    good = xi.xbus_frame(xi.MID_MTDATA2, mtdata2_payload(gyro=(1, 1, 1)))
    bad = bytearray(good)
    bad[-1] ^= 0xFF  # corrupt checksum
    p = xi.XbusParser()
    frames = p.feed(bytes(bad) + good)
    assert len(frames) == 1
    assert np.allclose(xi.parse_mtdata2(frames[0][1])["gyro"], [1, 1, 1],
                       atol=1e-6)


def test_extended_length_frame():
    payload = bytes(300)
    body = bytes([xi.BID, 0x42, 0xFF]) + struct.pack(">H", 300) + payload
    cksum = (-sum(body)) & 0xFF
    frame = bytes([xi.PREAMBLE]) + body + bytes([cksum])
    p = xi.XbusParser()
    frames = p.feed(frame)
    assert frames == [(0x42, payload)]


def test_precision_flag_bits_ignored_in_id_match():
    # low nibble of the data id carries format flags; parser matches on the
    # masked id
    body = struct.pack(">HB3f", xi.XDI_RATE_OF_TURN | 0x0004, 12, 7, 8, 9)
    fields = xi.parse_mtdata2(body)
    assert np.allclose(fields["gyro"], [7, 8, 9], atol=1e-5)


def test_mount_rotation_math():
    # 90 deg yaw mount: sensor x points to robot's left
    m = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    v = np.array([1.0, 0, 0])
    out = xi._rotate(m, v)
    assert np.allclose(out, [0, 1, 0], atol=1e-9)
    q = np.array([1.0, 0, 0, 0])
    qb = xi._quat_mul(q, xi._quat_conj(m))
    # base-frame quat should encode the -90 yaw of base relative to sensor
    assert abs(np.linalg.norm(qb) - 1) < 1e-9


def test_double_precision_and_flags():
    q = (0.7071, 0.7071, 0.0, 0.0)
    body = struct.pack(">HB4d", xi.XDI_QUATERNION | 0x3, 32, *q)
    fields = xi.parse_mtdata2(body)
    assert np.allclose(fields["quat"], q, atol=1e-9)
    assert fields["coord"] == 0


def test_fixed_point_1220():
    vals = (0.5, -1.25, 2.0)
    raw = struct.pack(">HB3i", xi.XDI_RATE_OF_TURN | 0x1, 12,
                      *(int(v * (1 << 20)) for v in vals))
    fields = xi.parse_mtdata2(raw)
    assert np.allclose(fields["gyro"], vals, atol=1e-5)


def test_ned_coordinate_flag_reported():
    q = (1.0, 0.0, 0.0, 0.0)
    body = struct.pack(">HB4f", xi.XDI_QUATERNION | 0x4, 16, *q)  # NED flag
    fields = xi.parse_mtdata2(body)
    assert fields["coord"] == 1
