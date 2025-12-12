# UDP_header.py
import struct
import json
from enum import IntEnum

# Packet Type Enum
class PacketType(IntEnum):
    CONTROL = 0
    DATA = 1
    ACK = 2

# Header format:
# ! - big endian
# B - packet_type (1 byte)
# I - seq_num (4 bytes)
# I - frame_id (4 bytes)
# H - frag_index (2 bytes)
# H - total_frags (2 bytes)
# d - timestamp double (8 bytes)
# B - is_high_res (1 byte)
HEADER_FORMAT = "!BIIHHdB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class UDPHeader:
    def __init__(self, packet_type, seq_num,
                 frame_id, frag_index, total_frags,
                 timestamp, is_high_res):
        self.packet_type = packet_type
        self.seq_num = seq_num
        self.frame_id = frame_id
        self.frag_index = frag_index
        self.total_frags = total_frags
        self.timestamp = timestamp
        self.is_high_res = is_high_res

    def pack(self):
        return struct.pack(
            HEADER_FORMAT,
            int(self.packet_type),
            self.seq_num,
            self.frame_id,
            self.frag_index,
            self.total_frags,
            float(self.timestamp),
            1 if self.is_high_res else 0
        )

    @staticmethod
    def unpack(data):
        unpacked = struct.unpack(HEADER_FORMAT, data)
        return UDPHeader(
            packet_type=PacketType(unpacked[0]),
            seq_num=unpacked[1],
            frame_id=unpacked[2],
            frag_index=unpacked[3],
            total_frags=unpacked[4],
            timestamp=unpacked[5],
            is_high_res=bool(unpacked[6])
        )


def encode_packet(header: UDPHeader, payload: bytes):
    return header.pack() + payload


def decode_packet(packet: bytes):
    header_bytes = packet[:HEADER_SIZE]
    payload = packet[HEADER_SIZE:]
    header = UDPHeader.unpack(header_bytes)
    return header, payload
