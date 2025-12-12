import socket
import json
import time
import threading
import queue

import cv2
import numpy as np

from UDP_header import UDPHeader, PacketType, encode_packet, decode_packet

# =======================
# Network configuration
# =======================
SERVER_IP = "127.0.0.1"
SERVER_PORT = 50000
CLIENT_PORT = 50001

MAX_UDP_SIZE = 65507
BUFFER_SIZE = 30   # how many frames we allow queued on the client

# =======================
# Try to import torch (AI upscaler)
# =======================
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    print("[AI] PyTorch imported successfully, FSRCNN enabled.")
except Exception as e:
    print("[AI] Could not import torch, AI upscaler disabled. Using bicubic only.")
    print(f"[AI] Import error: {e}")
    TORCH_AVAILABLE = False


USE_AI_UPSCALER = True  # you can set this to False to force bicubic


# =======================
# FSRCNN Super-Resolution (only if torch is available)
# =======================
if TORCH_AVAILABLE:

    class FSRCNN(torch.nn.Module):
        """
        A simple FSRCNN-like architecture for 2x super-resolution.
        This is a minimal version suitable for your project structure.
        """

        def __init__(self, scale_factor=2, d=56, s=12, m=4):
            super().__init__()
            self.scale_factor = scale_factor

            # Feature extraction
            self.feature_extraction = torch.nn.Conv2d(3, d, kernel_size=5, padding=2)

            # Shrinking
            self.shrink = torch.nn.Conv2d(d, s, kernel_size=1)

            # Mapping (m conv layers)
            mapping_layers = []
            for _ in range(m):
                mapping_layers.append(torch.nn.Conv2d(s, s, kernel_size=3, padding=1))
                mapping_layers.append(torch.nn.ReLU(inplace=True))
            self.mapping = torch.nn.Sequential(*mapping_layers)

            # Expanding
            self.expand = torch.nn.Conv2d(s, d, kernel_size=1)

            # Deconvolution (upsampling)
            self.deconv = torch.nn.ConvTranspose2d(
                d,
                3,
                kernel_size=9,
                stride=scale_factor,
                padding=4,
                output_padding=scale_factor - 1
            )

            self.relu = torch.nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(self.feature_extraction(x))
            x = self.relu(self.shrink(x))
            x = self.mapping(x)
            x = self.relu(self.expand(x))
            x = self.deconv(x)
            return x

    # Device + global FSRCNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fsrcnn_model = FSRCNN(scale_factor=2).to(device)
    fsrcnn_model.eval()

    # If you ever get a real pretrained checkpoint, you can load it here:
    # fsrcnn_model.load_state_dict(torch.load("fsrcnn_x2.pth", map_location=device)


def ai_upscale_to_1080(frame: np.ndarray) -> np.ndarray:
    """
    If possible:
      - run FSRCNN (2x SR) and then resize to 1920x1080 if needed.
    Otherwise:
      - fallback to bicubic interpolation.

    This function NEVER crashes the client: any error -> bicubic.
    """

    # If AI disabled globally or torch is not available → bicubic
    if not USE_AI_UPSCALER or not TORCH_AVAILABLE:
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)

    try:
        # Convert BGR uint8 [H, W, C] to float32 tensor [1, C, H, W] in [0, 1]
        img = frame.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = fsrcnn_model(img_tensor)

        sr_tensor = torch.clamp(sr_tensor, 0.0, 1.0)
        sr_img = sr_tensor.squeeze(0).cpu().numpy()  # [C, H, W]
        sr_img = np.transpose(sr_img, (1, 2, 0))      # CHW -> HWC
        sr_img = (sr_img * 255.0).astype(np.uint8)

        # If size is not exactly 1920x1080, resize
        h, w, _ = sr_img.shape
        if (w, h) != (1920, 1080):
            sr_img = cv2.resize(sr_img, (1920, 1080), interpolation=cv2.INTER_CUBIC)

        print("[AI] FSRCNN ran successfully on low-res frame.")
        return sr_img

    except Exception as e:
        print(f"[AI] Exception in FSRCNN upscaler: {e}")
        print("[AI] Falling back to bicubic upscale for this frame.")
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_CUBIC)


# =======================
# Frame Reassembler
# =======================
class FrameReassembler:
    """
    Reassembles a JPEG frame from multiple UDP fragments.
    frames[fid] = list of fragments (bytes or None)
    """

    def __init__(self):
        # frame_id -> list[bytes|None]
        self.frames = {}

    def add_fragment(self, hdr: UDPHeader, payload: bytes):
        fid = hdr.frame_id
        if fid not in self.frames:
            self.frames[fid] = [None] * hdr.total_frags
        self.frames[fid][hdr.frag_index] = payload

        # If all fragments are present, join them and return full JPEG bytes
        if all(frag is not None for frag in self.frames[fid]):
            frame_data = b"".join(self.frames[fid])
            del self.frames[fid]
            return frame_data
        return None


# =======================
# Display Thread
# =======================
def display_thread(frame_queue: "queue.Queue", stop_event: threading.Event):
    """
    Display at a fixed 24 FPS, regardless of how fast frames arrive.
    Also prints which frame_id is actually being displayed.
    """
    FRAME_INTERVAL = 1.0 / 24.0  # 24 FPS = ~41.67 ms per frame
    last_time = time.time()

    last_frame = None

    while not stop_event.is_set():
        now = time.time()
        elapsed = now - last_time

        if elapsed >= FRAME_INTERVAL:
            # Try to get a new frame without blocking; if none, keep the old one
            try:
                frame_id, frame, is_high = frame_queue.get_nowait()

                if frame is None:
                    # Sentinel: end of stream / shutdown
                    break

                # High-res → show directly; low-res → AI upscaler
                if is_high:
                    last_frame = frame
                    res_str = "1080p"
                else:
                    last_frame = ai_upscale_to_1080(frame)
                    res_str = "480p -> AI(SR) -> 1080p"

                print(f"[display] showing frame {frame_id} ({res_str})")

            except queue.Empty:
                # No new frame available; we'll keep showing the last one
                pass

            if last_frame is not None:
                cv2.imshow("UDP Video", last_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

            last_time = now
        else:
            # Sleep the remaining time so we don't run too fast
            time.sleep(FRAME_INTERVAL - elapsed)

    cv2.destroyAllWindows()


# =======================
# Feedback Thread
# =======================
def feedback_thread(
    sock: socket.socket,
    frame_queue: "queue.Queue",
    stop_event: threading.Event,
    stats: dict,
):
    """
    Periodically send FEEDBACK control packets to the server.

    Payload JSON:
    {
        "cmd": "FEEDBACK",
        "buffer_level": <frames currently in queue>,
        "buffer_capacity": BUFFER_SIZE,
        "accept_rate": <frames/sec queued in last interval>
    }
    """

    FEEDBACK_INTERVAL = 0.5  # seconds

    prev_time = time.time()
    prev_frames_done = 0

    while not stop_event.is_set():
        time.sleep(FEEDBACK_INTERVAL)
        now = time.time()
        frames_done = stats["frames_completed"]

        dt = now - prev_time
        df = frames_done - prev_frames_done
        accept_rate = (df / dt) if dt > 0 else 0.0

        prev_time = now
        prev_frames_done = frames_done

        buffer_level = frame_queue.qsize()

        msg = {
            "cmd": "FEEDBACK",
            "buffer_level": buffer_level,
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
            sock.sendto(
                encode_packet(hdr, json.dumps(msg).encode("utf-8")),
                (SERVER_IP, SERVER_PORT),
            )
        except OSError:
            break  # socket closed


# =======================
# Main Client Logic
# =======================
def run_client(video_name="Video", desired_quality="high"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", CLIENT_PORT))
    sock.settimeout(5.0)

    # ----- Send REQUEST_VIDEO CONTROL packet -----
    control_msg = {
        "cmd": "REQUEST_VIDEO",
        "video": video_name,
        "quality": desired_quality,
    }

    hdr = UDPHeader(
        packet_type=PacketType.CONTROL,
        seq_num=0,
        frame_id=0,
        frag_index=0,
        total_frags=1,
        timestamp=int(time.time() * 1e9),
        is_high_res=True,
    )

    sock.sendto(
        encode_packet(hdr, json.dumps(control_msg).encode("utf-8")),
        (SERVER_IP, SERVER_PORT),
    )
    print(f"[INFO] Sent REQUEST_VIDEO for '{video_name}' quality={desired_quality}")

    # ----- Set up reassembly, display, feedback -----
    reassembler = FrameReassembler()
    frame_queue = queue.Queue(maxsize=BUFFER_SIZE)
    stop_event = threading.Event()

    # stats["frames_completed"] = how many frames successfully queued
    stats = {"frames_completed": 0}

    # Start display thread (24 FPS)
    disp_thread = threading.Thread(
        target=display_thread,
        args=(frame_queue, stop_event),
        daemon=True,
    )
    disp_thread.start()

    # Start feedback thread
    fb_thread = threading.Thread(
        target=feedback_thread,
        args=(sock, frame_queue, stop_event, stats),
        daemon=True,
    )
    fb_thread.start()

    try:
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(MAX_UDP_SIZE)
            except socket.timeout:
                continue

            try:
                hdr, payload = decode_packet(data)
            except Exception as e:
                print(f"[WARN] Failed to decode packet: {e}")
                continue

            # ===== DATA packet: a fragment of a frame =====
            if hdr.packet_type == PacketType.DATA:
                # Send ACK back (echo server timestamp)
                ack_hdr = UDPHeader(
                    packet_type=PacketType.ACK,
                    seq_num=0,
                    frame_id=hdr.frame_id,
                    frag_index=hdr.frag_index,
                    total_frags=hdr.total_frags,
                    timestamp=hdr.timestamp,
                    is_high_res=hdr.is_high_res,
                )
                sock.sendto(encode_packet(ack_hdr, b""), addr)

                frame_data = reassembler.add_fragment(hdr, payload)
                if frame_data:
                    frame = cv2.imdecode(
                        np.frombuffer(frame_data, np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is not None:
                        try:
                            # Queue: (frame_id, frame, is_high_res)
                            frame_queue.put_nowait(
                                (hdr.frame_id, frame, hdr.is_high_res)
                            )
                            stats["frames_completed"] += 1
                            print(
                                f"[client] frame {hdr.frame_id} queued, "
                                f"buffer={frame_queue.qsize()} frames"
                            )
                        except queue.Full:
                            print(
                                f"[WARN] Frame buffer full, "
                                f"dropping frame {hdr.frame_id}"
                            )

            # ===== CONTROL packet: check for END_OF_STREAM =====
            elif hdr.packet_type == PacketType.CONTROL:
                try:
                    msg = json.loads(payload.decode("utf-8"))
                except Exception:
                    msg = {}

                if msg.get("cmd") == "END_OF_STREAM":
                    print("[INFO] Received END_OF_STREAM from server")
                    stop_event.set()
                    # Tell display_thread to exit
                    try:
                        frame_queue.put_nowait((None, None, None))
                    except queue.Full:
                        pass
                    break

    finally:
        stop_event.set()
        try:
            frame_queue.put_nowait((None, None, None))
        except queue.Full:
            pass

        sock.close()
        print("[INFO] Client socket closed.")


if __name__ == "__main__":
    run_client()
