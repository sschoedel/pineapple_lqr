"""Laptop-side operator console for the Pineapple V3 LQR runner.

Run this ON YOUR LAPTOP (not on the Pi):

    uv run python deploy_lqr_console.py <pi-address> [port]

It sends a heartbeat to the runner 10x per second; the runner's deadman
DAMPS THE ROBOT if heartbeats stop for deadman_timeout (default 0.5 s) —
so losing wifi/ssh/this process = robot goes limp, by design. The console
prints loud warnings the moment the link degrades.

Commands: stand | balance | sit | damp | v <vx> <wz> | zero | exit
Ctrl-C sends 'damp' before exiting.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time

HB_PERIOD = 0.1  # 10 Hz heartbeats
LINK_WARN_AFTER = 0.3  # warn if no runner response for this long

LINK_LOST_BANNER = """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  LINK TO ROBOT LOST ({age:4.1f}s without response)                   !!
!!  the runner's deadman DAMPS the robot 0.5s after heartbeats stop !!
!!  it will stay limp until you reconnect and re-arm a mode         !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

TRIPPED_BANNER = """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  ROBOT SAFETY TRIP: {reason:<45s}!!
!!  motors are in damp mode; send a mode command to re-arm          !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""


def main(host: str, port: int = 43613) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.2)
    addr = (host, port)
    lock = threading.Lock()
    pending: list = []  # commands to piggyback on the next heartbeat
    running = {"on": True}
    state = {"last_reply": 0.0, "seq": 0, "warned": False, "trip_shown": None}

    def hb_loop():
        while running["on"]:
            with lock:
                cmd = pending.pop(0) if pending else None
            state["seq"] += 1
            pkt: dict = {"hb": state["seq"]}
            if cmd is not None:
                pkt["cmd"] = cmd
            try:
                sock.sendto(json.dumps(pkt).encode(), addr)
                data, _ = sock.recvfrom(1024)
                reply = json.loads(data.decode())
                state["last_reply"] = time.monotonic()
                if state["warned"]:
                    state["warned"] = False
                    print("\n== link to robot RESTORED ==")
                    print(f"   mode={reply['mode']} tripped={reply['tripped']}")
                if reply.get("tripped") and reply.get("reason") != state["trip_shown"]:
                    state["trip_shown"] = reply.get("reason")
                    print(TRIPPED_BANNER.format(reason=str(reply.get("reason"))))
                if not reply.get("tripped"):
                    state["trip_shown"] = None
                if cmd is not None:
                    print(f"   ack: mode={reply['mode']} v={reply['v']:+.2f} w={reply['w']:+.2f}")
            except (TimeoutError, OSError, ValueError):
                pass
            age = time.monotonic() - state["last_reply"]
            if age > LINK_WARN_AFTER and state["last_reply"] > 0 and not state["warned"]:
                state["warned"] = True
                print(LINK_LOST_BANNER.format(age=age))
            elif state["warned"] and age > LINK_WARN_AFTER:
                # keep shouting once a second while the link is down
                if int(age) != int(age - HB_PERIOD):
                    print(LINK_LOST_BANNER.format(age=age))
            time.sleep(HB_PERIOD)

    thread = threading.Thread(target=hb_loop, daemon=True)
    thread.start()
    print(f"console -> {host}:{port} | heartbeats at {1/HB_PERIOD:.0f} Hz")
    print("commands: stand | balance | sit | damp | v <vx> <wz> | zero | exit")

    try:
        while True:
            line = input("LQR> ").strip().split()
            if not line:
                continue
            if line[0] == "exit":
                break
            if line[0] == "v" and len(line) == 3:
                with lock:
                    pending.append(["v", float(line[1]), float(line[2])])
            elif line[0] in ("stand", "balance", "sit", "damp", "zero"):
                with lock:
                    pending.append([line[0]])
            else:
                print("unknown command")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        # park the robot before leaving
        with lock:
            pending.append(["damp"])
        time.sleep(3 * HB_PERIOD)
        running["on"] = False
        thread.join(timeout=1.0)
        print("\nconsole closed (sent damp). Runner deadman keeps the robot")
        print("limp until a console reconnects and re-arms a mode.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python deploy_lqr_console.py <pi-address> [port]")
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 43613)
