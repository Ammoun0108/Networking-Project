# udp_server_flowcontrol.py
#
# UDP video server with:
# - Fragmented JPEG frame streaming
# - Adaptive 1080p / 480p based on RTT-derived throughput from ACKs
# - Optional client FEEDBACK (buffer_level, accept_rate) to refine decisions
# - Loops video 5 times, then sends END_OF_STREAM and closes
#
# Directories expected:
#   library/1080p/frame_XXXX.jpg
#   library/480p/frame_XXXX.jpg

import socket
import time
import json
from pathlib import Path
from math import ceil

from UDP_header import UDPHeader, PacketType, encode_packet, decode_packet

# ===== Configuration =====
SERVER_IP = "0.0.0.0"
SERVER_PORT = 50000
FPS = 24
CHUNK_PAYLOAD_MAX = 1200

SOCK_HANDSHAKE_TIMEOUT = 0.5  # seconds for initial REQUEST_VIDEO
NUM_LOOPS = 5                 # loop the video this many times

LIB_ROOT = Path(__file__).parent / "Library"
DIR_1080 = LIB_ROOT / "1080p"
DIR_480 = LIB_ROOT / "480p"

# ===== Throughput-based flow control (RTT + frame size) =====
alpha_throughput = 0.1
smoothed_throughput = None  # EMA of bytes/sec

# Throughput hysteresis thresholds (bytes/sec) – tune these
DOWN_SWITCH_THRESHOLD = 10e6   # 0.1 MB/s  ≈ 0.8 Mbps
UP_SWITCH_THRESHOLD   = 5e6  # 0.15 MB/s ≈ 1.2 Mbps


# Map frame_id -> frame size (bytes)
frame_sizes = {}

# ===== Feedback-based flow control =====
# Client sends FEEDBACK with:
#   - buffer_level (frames queued on client)
#   - buffer_capacity (max frames)
#   - accept_rate (frames/sec successfully queued)
FEEDBACK_STALE_SEC = 2.0  # ignore feedback older than this


def list_frames(dir_path: Path):
    return sorted(dir_path.glob("frame_*.jpg"))


def load_bytes(path: Path) -> bytes:
    return path.read_bytes()


def make_chunks(buf: bytes, chunk_size: int):
    total = max(1, ceil(len(buf) / chunk_size))
    for i in range(total):
        off = i * chunk_size
        yield i, total, buf[off:off + chunk_size]


def process_ack_from_client(hdr: UDPHeader):
    """
    Use ACK timestamp + current time to measure RTT for that frame
    and update global smoothed_throughput based on frame size.

    Throughput = frame_size / RTT
    """
    global smoothed_throughput, alpha_throughput, frame_sizes

    # Only use ACK for the LAST fragment of a frame
    if hdr.frag_index != hdr.total_frags - 1:
        return

    now = time.monotonic()
    rtt_sec = now - hdr.timestamp  # hdr.timestamp was set when frame was sent

    if rtt_sec <= 0:
        return

    size = frame_sizes.get(hdr.frame_id)
    if size is None:
        return

    inst_tp = size / rtt_sec  # bytes/sec

    if smoothed_throughput is None:
        smoothed_throughput = inst_tp
    else:
        smoothed_throughput = (
            alpha_throughput * inst_tp
            + (1.0 - alpha_throughput) * smoothed_throughput
        )

    print(
        f"[RTT] frame={hdr.frame_id} size={size}B "
        f"RTT={rtt_sec*1000:.1f} ms "
        f"inst={inst_tp/1e6:.2f} MB/s "
        f"smoothed={smoothed_throughput/1e6:.2f} MB/s"
    )


def drain_control_packets(sock: socket.socket, client_addr, feedback_state: dict):
    """
    Non-blocking drain of incoming packets while streaming.
    We care about:
      - CONTROL packets with cmd == "FEEDBACK"
      - ACK packets for RTT / throughput
    """
    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except BlockingIOError:
            # No more data available
            break
        except socket.error:
            # Some other socket issue; treat as no more data
            break

        # Ignore packets from other addresses
        if client_addr is not None and addr != client_addr:
            continue

        if not data:
            continue

        try:
            hdr, payload = decode_packet(data)
        except Exception:
            continue

        # ===== CONTROL: FEEDBACK =====
        if hdr.packet_type == PacketType.CONTROL:
            try:
                msg = json.loads(payload.decode("utf-8"))
            except Exception:
                msg = {}

            if msg.get("cmd") == "FEEDBACK":
                feedback_state["buffer_level"] = msg.get("buffer_level", 0)
                feedback_state["buffer_capacity"] = msg.get("buffer_capacity", None)
                feedback_state["accept_rate"] = msg.get("accept_rate", 0.0)
                feedback_state["last_update"] = time.monotonic()

        # ===== ACK: use RTT + frame size to update throughput =====
        elif hdr.packet_type == PacketType.ACK:
            process_ack_from_client(hdr)


def choose_resolution(smoothed_tp,
                      last_is_high: bool,
                      feedback_state: dict):
    """
    Decide whether to send 1080p (True) or 480p (False) using:
      - RTT-based throughput hysteresis
      - receiver feedback (buffer_level, accept_rate) as a modifier

    Returns (is_high, last_is_high_updated).
    """

    # ----- Base decision: throughput hysteresis from RTT -----
    if smoothed_tp is None:
        # At the beginning, be optimistic and start high-res
        is_high = True
    else:
        if smoothed_tp < DOWN_SWITCH_THRESHOLD:
            last_is_high = False
        elif smoothed_tp > UP_SWITCH_THRESHOLD:
            last_is_high = True
        is_high = last_is_high

    # ----- Modify decision using feedback, if recent -----
    now = time.monotonic()
    last_fb_time = feedback_state.get("last_update", 0.0)
    buffer_capacity = feedback_state.get("buffer_capacity", None)

    if buffer_capacity is not None and (now - last_fb_time) <= FEEDBACK_STALE_SEC:
        buffer_level = feedback_state.get("buffer_level", 0)
        accept_rate = feedback_state.get("accept_rate", 0.0)

        high_mark = 0.8 * buffer_capacity   # near-full buffer
        low_mark = 0.3 * buffer_capacity    # fairly empty buffer

        # If client is overloaded: buffer almost full OR can't keep up with FPS
        if buffer_level >= high_mark or accept_rate < FPS * 0.8:
            is_high = False
            last_is_high = False

        # If client is very comfortable: buffer low AND draining at >= FPS
        elif buffer_level <= low_mark and accept_rate >= FPS * 0.9:
            is_high = True
            last_is_high = True

    return is_high, last_is_high


def send_end_of_stream(sock: socket.socket, client_addr):
    """Send END_OF_STREAM control packet to client."""
    hdr = UDPHeader(
        packet_type=PacketType.CONTROL,
        seq_num=0,
        frame_id=0,
        frag_index=0,
        total_frags=1,
        timestamp=time.monotonic(),
        is_high_res=True,
    )
    payload = json.dumps({"cmd": "END_OF_STREAM"}).encode("utf-8")
    sock.sendto(encode_packet(hdr, payload), client_addr)
    print("[server] Sent END_OF_STREAM")


# ===== Main =====
def main():
    global smoothed_throughput, frame_sizes

    frames_1080 = list_frames(DIR_1080)
    frames_480 = list_frames(DIR_480)

    n1080 = len(frames_1080)
    n480 = len(frames_480)

    if not n1080 and not n480:
        raise RuntimeError("No frames found in 1080p or 480p directories.")

    num_frames = max(n1080, n480)
    print(f"[server] Using {num_frames} logical frames (1080p: {n1080}, 480p: {n480})")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((SERVER_IP, SERVER_PORT))
    sock.settimeout(SOCK_HANDSHAKE_TIMEOUT)

    print(f"[server] Listening on {SERVER_IP}:{SERVER_PORT} ...")

    client_addr = None

    # ===== Wait for REQUEST_VIDEO =====
    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue

        print(f"[server] got packet len={len(data)} from {addr}")

        if not data:
            continue

        try:
            hdr, payload = decode_packet(data)
        except Exception as e:
            print(f"[server] decode_packet failed in handshake: {e}")
            continue

        if hdr.packet_type != PacketType.CONTROL:
            # Expect CONTROL packet for handshake
            continue

        try:
            msg = json.loads(payload.decode("utf-8"))
        except Exception as e:
            print(f"[server] JSON decode failed in handshake: {e}")
            continue

        print(f"[server] CONTROL msg in handshake: {msg}")

        if msg.get("cmd") == "REQUEST_VIDEO":
            client_addr = addr
            print(f"[server] Got REQUEST_VIDEO from {client_addr}")
            break

    # After handshake, switch to non-blocking mode so we can
    # send frames at FPS but still occasionally read FEEDBACK/ACK.
    sock.settimeout(0.0)
    sock.setblocking(False)

    t0 = time.perf_counter()
    interval = 1.0 / FPS

    # Per-24-frame batch counters
    batch_1080_count = 0
    batch_480_count = 0

    # Track last chosen resolution so hysteresis works
    last_is_high = True

    # Feedback state from client
    feedback_state = {
        "buffer_level": 0,
        "buffer_capacity": None,
        "accept_rate": 0.0,
        "last_update": 0.0,
    }

    print("[server] Start streaming ...")

    try:
        for loop_idx in range(NUM_LOOPS):
            print(f"[server] ===== LOOP {loop_idx+1}/{NUM_LOOPS} =====")

            frame_idx = 0
            frame_id_loop_offset = loop_idx * num_frames

            while frame_idx < num_frames:

                frame_id = frame_id_loop_offset + frame_idx + 1

                # 1) Drain FEEDBACK + ACKs
                drain_control_packets(sock, client_addr, feedback_state)

                # 2) Decide resolution (1080p vs 480p)
                is_high, last_is_high = choose_resolution(
                    smoothed_throughput,
                    last_is_high,
                    feedback_state
                )

                # 3) Pick frame path
                has_1080 = frame_idx < n1080
                has_480 = frame_idx < n480

                if is_high and has_1080:
                    path = frames_1080[frame_idx]
                    is_high = True
                elif has_480:
                    path = frames_480[frame_idx]
                    is_high = False
                else:
                    # fallback
                    path = frames_1080[frame_idx]
                    is_high = True

                # 4) Read JPEG
                jpeg = load_bytes(path)

                # 5) Store size for RTT throughput measurement
                frame_sizes[frame_id] = len(jpeg)

                # Timestamp for RTT (monotonic, seconds as float)
                frame_ts = time.monotonic()

                # 6) Send all fragments
                chunks = list(make_chunks(jpeg, CHUNK_PAYLOAD_MAX))
                for frag_index, total_frags, payload in chunks:
                    hdr = UDPHeader(
                        packet_type=PacketType.DATA,
                        seq_num=0,
                        frame_id=frame_id,
                        frag_index=frag_index,
                        total_frags=total_frags,
                        timestamp=frame_ts,
                        is_high_res=is_high,
                    )
                    pkt = encode_packet(hdr, payload)
                    sock.sendto(pkt, client_addr)

                # 7) Log every 24 frames
                if frame_id % 24 == 0:
                    tp_str = (
                        f"{smoothed_throughput/1e6:.2f} MB/s"
                        if smoothed_throughput is not None
                        else "None"
                    )
                    fb_cap = feedback_state.get("buffer_capacity")
                    fb_lvl = feedback_state.get("buffer_level")
                    fb_acc = feedback_state.get("accept_rate")
                    print(
                        f"[server] frame {frame_id}, "
                        f"res={'1080p' if is_high else '480p'}, "
                        f"bytes={len(jpeg)}, "
                        f"smoothed_throughput={tp_str}, "
                        f"batch_1080={batch_1080_count}, "
                        f"batch_480={batch_480_count}, "
                        f"feedback: buffer={fb_lvl}/{fb_cap}, "
                        f"accept_rate={fb_acc:.2f} fps"
                    )
                    batch_1080_count = 0
                    batch_480_count = 0

                # Count for batches
                if is_high:
                    batch_1080_count += 1
                else:
                    batch_480_count += 1

                # 8) FPS pacing
                next_deadline = t0 + frame_id * interval
                now = time.perf_counter()
                if now < next_deadline:
                    time.sleep(next_deadline - now)

                frame_idx += 1

        # After finishing ALL loops:
        print(f"[server] Completed {NUM_LOOPS} loops. Sending END_OF_STREAM...")
        if client_addr is not None:
            send_end_of_stream(sock, client_addr)

    except KeyboardInterrupt:
        print("\n[server] Stopped by user.")
    finally:
        sock.close()
        print("[server] Closed.")


if __name__ == "__main__":
    main()
