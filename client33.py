import socket
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
