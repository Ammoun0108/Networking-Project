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
SOCK_TIMEOUT = 0.5  # used only during handshake

LIB_ROOT = Path(__file__).parent / "Library"
DIR_1080 = LIB_ROOT / "1080p"
DIR_480 = LIB_ROOT / "480p"

# Flow control
alpha_throughput = 0.1
smoothed_throughput = None  # bytes/sec

# Simple hysteresis thresholds (bytes per second)
DOWN_SWITCH_THRESHOLD = 2.5e8  # if below → use 480p
UP_SWITCH_THRESHOLD = 3.0e8    # if above → use 1080p


# ===== Helper functions =====
def list_frames(dir_path: Path):
    return sorted(dir_path.glob("frame_*.jpg"))


def load_bytes(path: Path) -> bytes:
    return path.read_bytes()


def make_chunks(buf: bytes, chunk_size: int):
    total = max(1, ceil(len(buf) / chunk_size))
    for i in range(total):
        off = i * chunk_size
        yield i, total, buf[off:off + chunk_size]


def drain_acks(sock, sent_frame_bytes):
    """
    Non-blocking: read all pending packets.
    When an ACK arrives, use its echoed timestamp to compute RTT and
    update smoothed_throughput based on frame_bytes / RTT.
    """
    global smoothed_throughput

    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except (BlockingIOError, OSError):
            # No more data available (non-blocking socket)
            break

        try:
            hdr, payload = decode_packet(data)
        except Exception:
            continue

        # ===== ACK packets → update RTT-based throughput =====
        if hdr.packet_type == PacketType.ACK:
            frame_id = hdr.frame_id
            frame_bytes = sent_frame_bytes.get(frame_id)

            # hdr.timestamp is the time (ns) when we sent that fragment.
            now_ns = time.monotonic_ns()
            rtt = (now_ns - hdr.timestamp) / 1e9  # seconds

            if frame_bytes and rtt > 0:
                inst_throughput = frame_bytes / rtt  # bytes/sec

                if smoothed_throughput is None:
                    smoothed_throughput = inst_throughput
                else:
                    smoothed_throughput = (
                        alpha_throughput * inst_throughput
                        + (1 - alpha_throughput) * smoothed_throughput
                    )

                print(
                    f"[server] ACK for frame {frame_id}: "
                    f"RTT={rtt:.4f}s, "
                    f"inst={inst_throughput/1e6:.2f} Mbps, "
                    f"smoothed={smoothed_throughput/1e6:.2f} Mbps"
                )

        # ===== CONTROL packets (e.g., FEEDBACK) – optional handling =====
        elif hdr.packet_type == PacketType.CONTROL:
            # You can optionally parse FEEDBACK here:
            # msg = json.loads(payload.decode("utf-8"))
            # print("[server] FEEDBACK:", msg)
            pass


# ===== Main =====
def main():
    global smoothed_throughput

    frames_1080 = list_frames(DIR_1080)
    frames_480 = list_frames(DIR_480)

    n1080 = len(frames_1080)
    n480 = len(frames_480)

    if not n1080 and not n480:
        raise RuntimeError("No frames found in 1080p or 480p directories.")

    num_frames = max(n1080, n480)
    print(f"[server] Using {num_frames} logical frames (1080p={n1080}, 480p={n480})")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((SERVER_IP, SERVER_PORT))
    sock.settimeout(SOCK_TIMEOUT)

    print(f"[server] Listening on {SERVER_IP}:{SERVER_PORT} ...")

    client_addr = None

    # ===== Wait for REQUEST_VIDEO =====
    print("[server] Waiting for REQUEST_VIDEO ...")
    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue

        try:
            hdr, payload = decode_packet(data)
        except Exception as e:
            print(f"[server] could not decode packet from {addr}: {e}")
            continue

        if hdr.packet_type != PacketType.CONTROL:
            continue

        try:
            msg = json.loads(payload.decode("utf-8"))
        except Exception as e:
            print(f"[server] Invalid JSON CONTROL message: {e}")
            continue

        print(f"[server] CONTROL packet from {addr}: {msg}")

        if msg.get("cmd") == "REQUEST_VIDEO":
            client_addr = addr
            print(f"[server] Got REQUEST_VIDEO from {client_addr}")
            break

    # After handshake → non-blocking (for ACK draining)
    sock.settimeout(0.0)

    frame_id = 1
    frame_idx = 0  # logical index 0..num_frames-1
    t0 = time.perf_counter()
    interval = 1.0 / FPS

    batch_1080_count = 0
    batch_480_count = 0
    last_is_high = True

    # Track bytes per frame_id for RTT-based throughput
    sent_frame_bytes = {}

    print("[server] Start streaming ...")

    try:
        while frame_idx < num_frames:

            # Decide resolution using hysteresis on smoothed_throughput
            if smoothed_throughput is None:
                is_high = True
            else:
                if smoothed_throughput < DOWN_SWITCH_THRESHOLD:
                    last_is_high = False
                elif smoothed_throughput > UP_SWITCH_THRESHOLD:
                    last_is_high = True
                is_high = last_is_high

            # Select frame for this logical index
            path = None
            has_1080 = frame_idx < n1080
            has_480 = frame_idx < n480

            if is_high and has_1080:
                path = frames_1080[frame_idx]
                is_high = True
                last_is_high = True
            elif has_480:
                path = frames_480[frame_idx]
                is_high = False
                last_is_high = False
            elif has_1080:
                path = frames_1080[frame_idx]
                is_high = True
                last_is_high = True
            else:
                print(f"[server] No frame available for index {frame_idx}, stopping.")
                break

            # Update per-batch counters
            if is_high:
                batch_1080_count += 1
            else:
                batch_480_count += 1

            jpeg = load_bytes(path)
            chunks = list(make_chunks(jpeg, CHUNK_PAYLOAD_MAX))
            frame_bytes = len(jpeg)
            sent_frame_bytes[frame_id] = frame_bytes

            # Send chunks for this frame
            for frag_index, total_frags, payload in chunks:
                ts_ns = time.monotonic_ns()
                hdr = UDPHeader(
                    packet_type=PacketType.DATA,
                    seq_num=0,
                    frame_id=frame_id,
                    frag_index=frag_index,
                    total_frags=total_frags,
                    timestamp=ts_ns,
                    is_high_res=is_high,
                )

                pkt = encode_packet(hdr, payload)
                sock.sendto(pkt, client_addr)

            # Drain any ACKs / FEEDBACK that arrived
            drain_acks(sock, sent_frame_bytes)

            # Per-24-frame log (uses latest smoothed_throughput)
            if frame_id % 24 == 0:
                st = smoothed_throughput or 0.0
                print(
                    f"[server] frame {frame_id}, "
                    f"res={'1080p' if is_high else '480p'}, "
                    f"bytes={frame_bytes}, "
                    f"smoothed_throughput={st/1e6:.2f} Mbps, "
                    f"batch_1080={batch_1080_count}, "
                    f"batch_480={batch_480_count}"
                )
                batch_1080_count = 0
                batch_480_count = 0

            # Pace to FPS
            next_deadline = t0 + frame_id * interval
            now = time.perf_counter()
            if now < next_deadline:
                time.sleep(next_deadline - now)

            frame_idx += 1
            frame_id += 1

        print(f"[server] Frames exhausted. Sent {frame_idx} logical frames.")

    except KeyboardInterrupt:
        print("\n[server] Interrupted by user.")
    finally:
        sock.close()
        print("[server] Closed.")


if __name__ == "__main__":
    main()
