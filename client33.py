import socket
import json
import time
import threading
import queue
import requests
import csv
import os
import random

import cv2
import numpy as np

from UDP_header import UDPHeader, PacketType, encode_packet, decode_packet

# =======================
# MODE SELECTION
# =======================
UPSCALE_MODE = "server"
# "local"   = FSRCNN (OpenCV dnn_superres, requires FSRCNN_x2.pb)
# "server"  = FastAPI HTTP upscaler
# "bicubic" = baseline resize

# =======================
# NETWORK EMULATION (NO SUDO REQUIRED)
# =======================
NETWORK_PROFILE = "moderate"
# "none"     : no jitter, no loss
# "moderate" : ~0–30ms jitter, low loss
# "bad"      : bigger jitter, more loss
# "burst"    : alternates good/bad every 10s (realistic)

# You can tune these if you want
PROFILE_CFG = {
    "none":     {"jitter_type": "none",   "loss": 0.0},
    "moderate": {"jitter_type": "uniform","jmin": 0.0,  "jmax": 0.030, "loss": 0.002},  # 0.2%
    "bad":      {"jitter_type": "uniform","jmin": 0.0,  "jmax": 0.080, "loss": 0.020},  # 2%
    "burst":    {"jitter_type": "burst",  "loss_good": 0.0, "loss_bad": 0.02},
}

# =======================
# Network
# =======================
SERVER_IP = "127.0.0.1"
SERVER_PORT = 50000
CLIENT_PORT = 50001

MAX_UDP_SIZE = 65507
BUFFER_SIZE = 30

# =======================
# FastAPI server config
# =======================
UPSCALE_SERVER = "http://127.0.0.1:8000/upscale"

# =======================
# Metrics Logger
# =======================
class MetricsLogger:
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.f = open(filename, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "t", "mode", "net_profile",
            "frame_id", "is_high", "queue",
            "lat_ms", "up_ms", "disp_fps",
            "dropped_total",
            "jitter_ms_last_packet",
            "loss_events_total"
        ])
        self.t0 = time.time()

    def log(self, mode, net_profile, frame_id, is_high, queue_sz,
            lat_ms, up_ms, disp_fps, dropped_total, jitter_ms, loss_total):
        self.w.writerow([
            time.time() - self.t0,
            mode, net_profile,
            frame_id,
            int(is_high),
            queue_sz,
            float(lat_ms),
            float(up_ms),
            float(disp_fps),
            int(dropped_total),
            float(jitter_ms),
            int(loss_total),
        ])
        self.f.flush()

    def close(self):
        self.f.close()


# =======================
# Local AI Upscaler
# =======================
class AIUpscaler:
    def __init__(self, model="FSRCNN_x2.pb"):
        self.enabled_local = (UPSCALE_MODE == "local")

        if not self.enabled_local:
            return

        if not hasattr(cv2, "dnn_superres"):
            raise RuntimeError("opencv-contrib required: pip install opencv-contrib-python")

        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model)
        self.sr.setModel("fsrcnn", 2)
        print(f"[AI] FSRCNN model loaded: {model}")

    def upscale(self, frame):
        if UPSCALE_MODE == "bicubic":
            return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)

        if UPSCALE_MODE == "local":
            try:
                up = self.sr.upsample(frame)  # 480->960
            except Exception as e:
                print(f"[AI] FSRCNN failed: {e} -> bicubic fallback")
                up = frame
            return cv2.resize(up, (1920, 1080), interpolation=cv2.INTER_CUBIC)

        if UPSCALE_MODE == "server":
            try:
                ok, enc = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError("cv2.imencode failed")

                r = requests.post(
                    UPSCALE_SERVER,
                    files={"image": enc.tobytes()},
                    timeout=0.4,
                )
                if r.status_code == 200:
                    arr = np.frombuffer(r.content, np.uint8)
                    out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if out is not None:
                        return out
            except Exception:
                pass

        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)


# =======================
# Frame Reassembler (timeout cleanup)
# =======================
class FrameReassembler:
    def __init__(self, timeout_sec=1.0):
        self.frames = {}
        self.times = {}
        self.timeout = float(timeout_sec)

    def cleanup(self):
        now = time.time()
        stale = [fid for fid, t0 in self.times.items() if (now - t0) > self.timeout]
        for fid in stale:
            self.frames.pop(fid, None)
            self.times.pop(fid, None)

    def add_fragment(self, hdr, payload):
        fid = hdr.frame_id

        if fid not in self.frames:
            self.frames[fid] = [None] * hdr.total_frags
            self.times[fid] = time.time()

        # bounds check
        if hdr.frag_index < 0 or hdr.frag_index >= hdr.total_frags:
            return None

        # store fragment
        self.frames[fid][hdr.frag_index] = payload

        if all(x is not None for x in self.frames[fid]):
            frame = b"".join(self.frames[fid])
            del self.frames[fid]
            del self.times[fid]
            return frame

        self.cleanup()
        return None


# =======================
# Network Emulator (jitter + loss)
# =======================
def emulate_network_delay_and_loss(profile_name, loss_total_ref, last_jitter_ms_ref):
    cfg = PROFILE_CFG.get(profile_name, PROFILE_CFG["none"])

    # Loss
    loss_p = 0.0
    jitter_s = 0.0

    if cfg.get("jitter_type") == "none":
        loss_p = cfg.get("loss", 0.0)
        jitter_s = 0.0

    elif cfg.get("jitter_type") == "uniform":
        loss_p = cfg.get("loss", 0.0)
        jitter_s = random.uniform(cfg["jmin"], cfg["jmax"])

    elif cfg.get("jitter_type") == "burst":
        # alternate good/bad every 10 seconds
        phase = int(time.time()) % 20
        bad = (phase >= 10)
        if bad:
            jitter_s = random.uniform(0.02, 0.10)   # 20–100ms
            loss_p = cfg.get("loss_bad", 0.02)
        else:
            jitter_s = random.uniform(0.0, 0.010)   # 0–10ms
            loss_p = cfg.get("loss_good", 0.0)

    # Apply loss
    if loss_p > 0 and random.random() < loss_p:
        loss_total_ref[0] += 1
        last_jitter_ms_ref[0] = jitter_s * 1000.0
        return False  # drop packet

    # Apply jitter delay
    if jitter_s > 0:
        time.sleep(jitter_s)

    last_jitter_ms_ref[0] = jitter_s * 1000.0
    return True  # keep packet


# =======================
# Display Thread (logs frame metrics)
# =======================
def display_thread(frame_queue, stop_event, upscaler, logger, dropped_total_ref, loss_total_ref, last_jitter_ms_ref):
    FRAME_INTERVAL = 1.0 / 24.0
    last_time = time.time()
    last_frame = None

    shown_times = []  # for rolling FPS

    while not stop_event.is_set():
        now = time.time()
        elapsed = now - last_time

        if elapsed >= FRAME_INTERVAL:
            try:
                frame_id, frame, is_high, recv_t = frame_queue.get_nowait()
                if frame is None:
                    break

                # Latency from frame completion to display
                lat_ms = (time.time() - recv_t) * 1000.0

                # Upscaling timing (only meaningful if low-res)
                t0 = time.perf_counter()
                if is_high:
                    last_frame = frame
                else:
                    last_frame = upscaler.upscale(frame)
                up_ms = (time.perf_counter() - t0) * 1000.0

                # Rolling FPS over last 2 seconds
                shown_times.append(time.time())
                while shown_times and (shown_times[-1] - shown_times[0]) > 2.0:
                    shown_times.pop(0)
                disp_fps = (len(shown_times) / 2.0) if len(shown_times) > 1 else 0.0

                # Log one row per displayed frame
                logger.log(
                    UPSCALE_MODE, NETWORK_PROFILE,
                    frame_id, is_high, frame_queue.qsize(),
                    lat_ms, up_ms, disp_fps,
                    dropped_total_ref[0],
                    last_jitter_ms_ref[0],
                    loss_total_ref[0]
                )

                print(f"[display] frame {frame_id} (lat={lat_ms:.1f}ms up={up_ms:.1f}ms q={frame_queue.qsize()})")

            except queue.Empty:
                pass

            if last_frame is not None:
                cv2.imshow("UDP Video", last_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            last_time = now
        else:
            time.sleep(max(0.0, FRAME_INTERVAL - elapsed))

    cv2.destroyAllWindows()


# =======================
# Feedback Thread
# =======================
def feedback_thread(sock, frame_queue, stop_event, stats):
    INTERVAL = 0.5
    prev_time = time.time()
    prev_frames = 0

    while not stop_event.is_set():
        time.sleep(INTERVAL)

        now = time.time()
        df = stats["frames_completed"] - prev_frames
        dt = now - prev_time
        accept_rate = (df / dt) if dt > 0 else 0.0

        prev_time = now
        prev_frames = stats["frames_completed"]

        msg = {
            "cmd": "FEEDBACK",
            "buffer_level": frame_queue.qsize(),
            "buffer_capacity": BUFFER_SIZE,
            "accept_rate": accept_rate,
        }

        hdr = UDPHeader(
            packet_type=PacketType.CONTROL,
            seq_num=0,
            frame_id=0,
            frag_index=0,
            total_frags=1,
            timestamp=int(now * 1e9),
            is_high_res=True,
        )

        try:
            sock.sendto(encode_packet(hdr, json.dumps(msg).encode("utf-8")), (SERVER_IP, SERVER_PORT))
        except OSError:
            break


# =======================
# Main
# =======================
def run_client(video="Video", quality="high"):
    # Metrics file per run
    metrics_path = f"results/metrics_{UPSCALE_MODE}_{NETWORK_PROFILE}.csv"
    logger = MetricsLogger(metrics_path)
    print(f"[metrics] writing -> {metrics_path}")

    # Mutable refs for counters shared across threads
    dropped_total_ref = [0]
    loss_total_ref = [0]
    last_jitter_ms_ref = [0.0]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", CLIENT_PORT))
    sock.settimeout(5.0)

    upscaler = AIUpscaler()

    # REQUEST_VIDEO
    req = {"cmd": "REQUEST_VIDEO", "video": video, "quality": quality}
    hdr = UDPHeader(
        packet_type=PacketType.CONTROL,
        seq_num=0,
        frame_id=0,
        frag_index=0,
        total_frags=1,
        timestamp=int(time.time() * 1e9),
        is_high_res=True,
    )
    sock.sendto(encode_packet(hdr, json.dumps(req).encode("utf-8")), (SERVER_IP, SERVER_PORT))
    print(f"[INFO] Sent REQUEST_VIDEO quality={quality}")

    reassembler = FrameReassembler(timeout_sec=1.0)
    frame_queue = queue.Queue(maxsize=BUFFER_SIZE)
    stop_event = threading.Event()
    stats = {"frames_completed": 0}

    threading.Thread(
        target=display_thread,
        args=(frame_queue, stop_event, upscaler, logger, dropped_total_ref, loss_total_ref, last_jitter_ms_ref),
        daemon=True,
    ).start()

    threading.Thread(
        target=feedback_thread,
        args=(sock, frame_queue, stop_event, stats),
        daemon=True,
    ).start()

    try:
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(MAX_UDP_SIZE)
            except socket.timeout:
                continue

            # ---- emulate jitter + loss here (packet-level) ----
            keep = emulate_network_delay_and_loss(NETWORK_PROFILE, loss_total_ref, last_jitter_ms_ref)
            if not keep:
                continue

            try:
                hdr, payload = decode_packet(data)
            except Exception:
                continue

            if hdr.packet_type == PacketType.DATA:
                frame_data = reassembler.add_fragment(hdr, payload)

                # ACK only when a frame is complete (1 ACK per frame)
                if frame_data:
                    ack_hdr = UDPHeader(
                        packet_type=PacketType.ACK,
                        seq_num=0,
                        frame_id=hdr.frame_id,
                        frag_index=hdr.total_frags - 1,
                        total_frags=hdr.total_frags,
                        timestamp=hdr.timestamp,   # echo server ts
                        is_high_res=hdr.is_high_res,
                    )
                    sock.sendto(encode_packet(ack_hdr, b""), addr)

                    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        try:
                            recv_t = time.time()
                            frame_queue.put_nowait((hdr.frame_id, frame, hdr.is_high_res, recv_t))
                            stats["frames_completed"] += 1
                        except queue.Full:
                            dropped_total_ref[0] += 1

            elif hdr.packet_type == PacketType.CONTROL:
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except Exception:
                    msg = {}

                if msg.get("cmd") == "END_OF_STREAM":
                    print("[INFO] END_OF_STREAM")
                    stop_event.set()
                    try:
                        frame_queue.put_nowait((None, None, None, 0.0))
                    except queue.Full:
                        pass
                    break

    finally:
        stop_event.set()
        try:
            frame_queue.put_nowait((None, None, None, 0.0))
        except queue.Full:
            pass
        sock.close()
        logger.close()
        print("[INFO] client closed")


if __name__ == "__main__":
    run_client()import socket
import json
import time
import threading
import queue
import requests

import cv2
import numpy as np

from UDP_header import UDPHeader, PacketType, encode_packet, decode_packet

# =======================
# MODE SELECTION
# =======================

UPSCALE_MODE = "local"  
# "local"   = FSRCNN
# "server"  = FastAPI
# "bicubic" = baseline


# =======================
# Network
# =======================

SERVER_IP = "127.0.0.1"
SERVER_PORT = 50000
CLIENT_PORT = 50001

MAX_UDP_SIZE = 65507
BUFFER_SIZE = 30


# =======================
# FastAPI server config
# =======================

UPSCALE_SERVER = "http://127.0.0.1:8000/upscale"


# =======================
# Local AI Upscaler
# =======================

class AIUpscaler:

    def __init__(self, model="FSRCNN_x2.pb"):
        if UPSCALE_MODE != "local":
            return

        if not hasattr(cv2, "dnn_superres"):
            raise RuntimeError(
                "opencv-contrib required: pip install opencv-contrib-python"
            )

        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model)
        self.sr.setModel("fsrcnn", 2)

        print("[AI] FSRCNN model loaded")

    def upscale(self, frame):

        if UPSCALE_MODE == "bicubic":
            return cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_CUBIC)

        if UPSCALE_MODE == "local":
            up = self.sr.upsample(frame)
            return cv2.resize(up,(1920,1080),interpolation=cv2.INTER_CUBIC)

        if UPSCALE_MODE == "server":

            _, enc = cv2.imencode(".jpg",frame)

            r = requests.post(
                UPSCALE_SERVER,
                files={"image":enc.tobytes()},
                timeout=0.3
            )

            if r.status_code == 200:

                arr = np.frombuffer(r.content,np.uint8)
                out = cv2.imdecode(arr,cv2.IMREAD_COLOR)

                if out is not None:
                    return out

        return cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_CUBIC)


# =======================
# Frame Reassembler
# =======================

class FrameReassembler:

    def __init__(self, timeout_sec=1.0):

        self.frames = {}
        self.times = {}
        self.timeout = timeout_sec

    def cleanup(self):

        now = time.time()

        remove = []

        for fid,t in self.times.items():

            if now - t > self.timeout:
                remove.append(fid)

        for fid in remove:

            del self.frames[fid]
            del self.times[fid]

    def add_fragment(self,hdr,payload):

        fid = hdr.frame_id

        if fid not in self.frames:

            self.frames[fid] = [None]*hdr.total_frags
            self.times[fid] = time.time()

        self.frames[fid][hdr.frag_index] = payload

        if all(f is not None for f in self.frames[fid]):

            frame = b"".join(self.frames[fid])

            del self.frames[fid]
            del self.times[fid]

            return frame

        self.cleanup()

        return None


# =======================
# Display Thread
# =======================

def display_thread(frame_queue, stop_event, upscaler):

    FRAME_INTERVAL = 1/24

    last_frame = None
    last_time = time.time()

    while not stop_event.is_set():

        now = time.time()

        if now - last_time >= FRAME_INTERVAL:

            try:

                frame_id, frame, is_high = frame_queue.get_nowait()

                if frame is None:
                    break

                if is_high:
                    last_frame = frame
                else:
                    last_frame = upscaler.upscale(frame)

                print(f"[display] frame {frame_id}")

            except queue.Empty:
                pass

            if last_frame is not None:

                cv2.imshow("UDP Video",last_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            last_time = now

        else:

            time.sleep(FRAME_INTERVAL-(now-last_time))

    cv2.destroyAllWindows()


# =======================
# Feedback Thread
# =======================

def feedback_thread(sock,frame_queue,stop_event,stats):

    INTERVAL = 0.5

    prev_time = time.time()
    prev_frames = 0

    while not stop_event.is_set():

        time.sleep(INTERVAL)

        now = time.time()

        df = stats["frames_completed"]-prev_frames
        dt = now-prev_time

        accept_rate = df/dt if dt>0 else 0

        prev_time = now
        prev_frames = stats["frames_completed"]

        msg = {

            "cmd":"FEEDBACK",
            "buffer_level":frame_queue.qsize(),
            "buffer_capacity":BUFFER_SIZE,
            "accept_rate":accept_rate
        }

        hdr = UDPHeader(
            packet_type=PacketType.CONTROL,
            seq_num=0,
            frame_id=0,
            frag_index=0,
            total_frags=1,
            timestamp=int(now*1e9),
            is_high_res=True
        )

        sock.sendto(
            encode_packet(hdr,json.dumps(msg).encode()),
            (SERVER_IP,SERVER_PORT)
        )


# =======================
# Main Client
# =======================

def run_client(video="Video",quality="high"):

    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    sock.bind(("0.0.0.0",CLIENT_PORT))

    sock.settimeout(5.0)

    upscaler = AIUpscaler()

    req = {

        "cmd":"REQUEST_VIDEO",
        "video":video,
        "quality":quality
    }

    hdr = UDPHeader(
        packet_type=PacketType.CONTROL,
        seq_num=0,
        frame_id=0,
        frag_index=0,
        total_frags=1,
        timestamp=int(time.time()*1e9),
        is_high_res=True
    )

    sock.sendto(
        encode_packet(hdr,json.dumps(req).encode()),
        (SERVER_IP,SERVER_PORT)
    )

    reassembler = FrameReassembler()

    frame_queue = queue.Queue(maxsize=BUFFER_SIZE)

    stop_event = threading.Event()

    stats = {"frames_completed":0}

    disp_thread = threading.Thread(
        target=display_thread,
        args=(frame_queue,stop_event,upscaler),
        daemon=True
    )

    disp_thread.start()

    fb_thread = threading.Thread(
        target=feedback_thread,
        args=(sock,frame_queue,stop_event,stats),
        daemon=True
    )

    fb_thread.start()

    try:

        while not stop_event.is_set():

            try:
                data,addr = sock.recvfrom(MAX_UDP_SIZE)
            except socket.timeout:
                continue

            try:
                hdr,payload = decode_packet(data)
            except:
                continue

            if hdr.packet_type == PacketType.DATA:

                frame_data = reassembler.add_fragment(hdr,payload)

                if frame_data:

                    ack_hdr = UDPHeader(
                        packet_type=PacketType.ACK,
                        seq_num=0,
                        frame_id=hdr.frame_id,
                        frag_index=hdr.total_frags-1,
                        total_frags=hdr.total_frags,
                        timestamp=hdr.timestamp,
                        is_high_res=hdr.is_high_res
                    )

                    sock.sendto(encode_packet(ack_hdr,b""),addr)

                    frame = cv2.imdecode(
                        np.frombuffer(frame_data,np.uint8),
                        cv2.IMREAD_COLOR
                    )

                    if frame is not None:

                        try:

                            frame_queue.put_nowait(
                                (hdr.frame_id,frame,hdr.is_high_res)
                            )

                            stats["frames_completed"]+=1

                        except queue.Full:

                            print("[WARN] buffer full")

            elif hdr.packet_type == PacketType.CONTROL:

                msg = json.loads(payload.decode())

                if msg.get("cmd") == "END_OF_STREAM":

                    stop_event.set()

                    frame_queue.put((None,None,None))

                    break

    finally:

        stop_event.set()

        sock.close()

        print("[INFO] client closed")


if __name__ == "__main__":
    run_client()
