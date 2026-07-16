"""Viser command center for the Pineapple V3 LQR runner (run ON THE LAPTOP).

    uv run python deploy_lqr_gui.py <pi-address> [--port 43613] [--gui-port 8080]

Opens a viser window (http://localhost:8080) with:
  - big E-STOP (damp) button, mode buttons (stand / balance / sit)
  - velocity & yaw-rate sliders with a SEND and a ZERO button
  - live state-estimator telemetry (tilt, forward velocity, yaw rate,
    wheel speeds, integrators) and link/trip status banners

It doubles as the operator deadman heartbeat (10 Hz over UDP): closing
this window / losing wifi / laptop sleep damps the robot within
deadman_timeout (0.5 s). Same protocol as deploy_lqr_console.py.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time

import numpy as np
import viser

HB_PERIOD = 0.1
LINK_LOST_AFTER = 0.4


class RunnerLink:
    """Heartbeat + command client for the runner's OperatorLink."""

    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.2)
        self.lock = threading.Lock()
        self.pending: list = []
        self.status: dict = {}
        self.last_reply = 0.0
        self.seq = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def send(self, cmd: list):
        with self.lock:
            self.pending.append(cmd)

    @property
    def link_age(self) -> float:
        return time.monotonic() - self.last_reply if self.last_reply else 1e9

    def _loop(self):
        while self.running:
            with self.lock:
                cmd = self.pending.pop(0) if self.pending else None
            self.seq += 1
            pkt: dict = {"hb": self.seq}
            if cmd is not None:
                pkt["cmd"] = cmd
            try:
                self.sock.sendto(json.dumps(pkt).encode(), self.addr)
                data, _ = self.sock.recvfrom(2048)
                self.status = json.loads(data.decode())
                self.last_reply = time.monotonic()
            except (TimeoutError, OSError, ValueError):
                pass
            time.sleep(HB_PERIOD)

    def close(self):
        self.send(["damp"])
        time.sleep(3 * HB_PERIOD)
        self.running = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host")
    ap.add_argument("--port", type=int, default=43613)
    ap.add_argument("--gui-port", type=int, default=8080)
    args = ap.parse_args()

    link = RunnerLink(args.host, args.port)
    server = viser.ViserServer(port=args.gui_port, label="Pineapple LQR")

    with server.gui.add_folder("SAFETY"):
        estop = server.gui.add_button("E-STOP (damp)", color="red")
        link_md = server.gui.add_markdown("**link:** connecting...")
        trip_md = server.gui.add_markdown("")

    with server.gui.add_folder("Mode"):
        b_stand = server.gui.add_button("stand")
        b_balance = server.gui.add_button("balance")
        b_sit = server.gui.add_button("sit")
        mode_md = server.gui.add_markdown("**mode:** ?")

    with server.gui.add_folder("Drive"):
        s_v = server.gui.add_slider("v (m/s)", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
        s_w = server.gui.add_slider("w (rad/s)", min=-2.0, max=2.0, step=0.05, initial_value=0.0)
        b_send = server.gui.add_button("send v/w")
        b_zero = server.gui.add_button("ZERO commands")

    with server.gui.add_folder("State estimator"):
        n_roll = server.gui.add_number("roll (rad)", initial_value=0.0, disabled=True)
        n_pitch = server.gui.add_number("pitch (rad)", initial_value=0.0, disabled=True)
        n_vx = server.gui.add_number("vx est (m/s)", initial_value=0.0, disabled=True)
        n_wz = server.gui.add_number("yaw rate (rad/s)", initial_value=0.0, disabled=True)
        n_z = server.gui.add_number("height err (m)", initial_value=0.0, disabled=True)
        n_wl = server.gui.add_number("wheel L (rad/s)", initial_value=0.0, disabled=True)
        n_wr = server.gui.add_number("wheel R (rad/s)", initial_value=0.0, disabled=True)
        integ_md = server.gui.add_markdown("integrators: -")

    estop.on_click(lambda _: link.send(["damp"]))
    b_stand.on_click(lambda _: link.send(["stand"]))
    b_balance.on_click(lambda _: link.send(["balance"]))
    b_sit.on_click(lambda _: link.send(["sit"]))
    b_send.on_click(lambda _: link.send(["v", float(s_v.value), float(s_w.value)]))

    def zero(_):
        s_v.value = 0.0
        s_w.value = 0.0
        link.send(["zero"])

    b_zero.on_click(zero)

    print(f"viser command center on http://localhost:{args.gui_port} -> {args.host}:{args.port}")
    try:
        while True:
            st = link.status
            age = link.link_age
            if age < LINK_LOST_AFTER:
                link_md.content = f"**link:** OK ({age*1e3:.0f} ms)"
            else:
                link_md.content = (
                    f"**link:** :red[LOST {age:.1f}s — robot self-damps after "
                    f"0.5 s and stays limp until re-armed]"
                )
            if st:
                mode_md.content = f"**mode:** {st.get('mode')}  |  v={st.get('v', 0):+.2f} w={st.get('w', 0):+.2f}"
                if st.get("tripped"):
                    trip_md.content = f"**:red[TRIPPED: {st.get('reason')}]** — press a mode button to re-arm"
                else:
                    trip_md.content = "trip: none"
                tel = st.get("telemetry") or {}
                if "roll" in tel:
                    n_roll.value = round(tel["roll"], 4)
                    n_pitch.value = round(tel["pitch"], 4)
                    n_vx.value = round(tel["vx"], 3)
                    n_wz.value = round(tel["dyaw"], 3)
                    n_z.value = round(tel["z"], 4)
                    n_wl.value = round(tel["wheel_l"], 2)
                    n_wr.value = round(tel["wheel_r"], 2)
                    ig = tel.get("integ", [])
                    if ig:
                        integ_md.content = (
                            f"integrators: v={ig[0]:+.3f} yaw={ig[1]:+.3f} roll={ig[2]:+.3f}"
                        )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()
        print("GUI closed (sent damp).")


if __name__ == "__main__":
    main()
