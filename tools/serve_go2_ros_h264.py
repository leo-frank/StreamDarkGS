#!/usr/bin/env python3
"""Relay GO2 /frontvideostream H.264 from onboard ROS 2 to one SSH client.

Bind only robot-local loopback. No ffmpeg, OpenCV or Python SDK is needed here.
Use SSH local forwarding from the field Ubuntu computer to reach this socket.
"""

import argparse
import re
import socket
import struct
import sys
import threading


START_CODE = re.compile(b'\x00\x00(?:\x00)?\x01')


def h264_packet(raw):
    if len(raw) < 24 or raw[:2] != b'\x00\x01':
        raise ValueError(f'unrecognized CDR header: {raw[:16].hex()}')
    _frame_time, resolution, size = struct.unpack_from('<QII', raw, 4)
    trailing = len(raw) - 20 - size
    if trailing < 0 or trailing > 3:
        raise ValueError(f'unrecognized video layout: resolution={resolution} size={size} bytes={len(raw)}')
    if resolution in (180, 360):
        return None
    if resolution != 720:
        raise ValueError(f'unrecognized video resolution: {resolution}')
    packet = raw[20:20 + size]
    if not START_CODE.match(packet):
        raise ValueError(f'expected Annex-B H.264 packet: {packet[:16].hex()}')
    return packet


def parameter_sets(packet):
    starts = list(START_CODE.finditer(packet))
    for index, match in enumerate(starts):
        header = match.end()
        end = starts[index + 1].start() if index + 1 < len(starts) else len(packet)
        if header < end and packet[header] & 0x1f in (7, 8):
            yield packet[match.start():end]


def has_idr(packet):
    return any(packet[match.end()] & 0x1f == 5
               for match in START_CODE.finditer(packet) if match.end() < len(packet))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18800)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be in 1..65535')

    import rclpy
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy
    from unitree_go.msg import Go2FrontVideoData

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', args.port))
    listener.listen(1)
    print(f'[source] listening on robot 127.0.0.1:{args.port}', flush=True)

    lock = threading.Lock()
    active = {'socket': None, 'packets': 0, 'skipped': 0, 'errors': 0}
    headers = {}
    gop = bytearray()
    stopping = threading.Event()

    def accept_clients():
        listener.settimeout(1)
        while not stopping.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with lock:
                old = active['socket']
                if old is not None:
                    old.close()
                active['socket'] = connection
                try:
                    if gop:
                        connection.sendall(gop)
                    else:
                        for nal_type in (7, 8):
                            if nal_type in headers:
                                connection.sendall(headers[nal_type])
                except OSError:
                    active['socket'] = None
                    connection.close()
                    continue
            print('[source] client connected', flush=True)

    accept_thread = threading.Thread(target=accept_clients, daemon=True)
    accept_thread.start()

    def on_video(raw):
        try:
            packet = h264_packet(raw)
            if packet is None:
                active['skipped'] += 1
                return
            with lock:
                for nal in parameter_sets(packet):
                    header = START_CODE.match(nal).end()
                    headers[nal[header] & 0x1f] = nal
                if has_idr(packet):
                    gop.clear()
                    for nal_type in (7, 8):
                        if nal_type in headers:
                            gop.extend(headers[nal_type])
                if gop or has_idr(packet):
                    gop.extend(packet)
                    if len(gop) > 8_000_000:
                        gop.clear()
                connection = active['socket']
                if connection is not None:
                    try:
                        connection.sendall(packet)
                    except OSError:
                        connection.close()
                        active['socket'] = None
                active['packets'] += 1
                if active['packets'] % 30 == 0:
                    print(f"[source] video720p={active['packets']} other_resolutions={active['skipped']} cached_gop_bytes={len(gop)}", flush=True)
        except ValueError as exc:
            active['errors'] += 1
            if active['errors'] <= 3:
                print(f'[source-warning] {exc}', file=sys.stderr, flush=True)

    rclpy.init()
    node = rclpy.create_node('go2_ros_h264_source')
    qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    subscription = node.create_subscription(Go2FrontVideoData, '/frontvideostream', on_video, qos, raw=True)
    try:
        rclpy.spin(node)
    finally:
        stopping.set()
        listener.close()
        with lock:
            if active['socket'] is not None:
                active['socket'].close()
        accept_thread.join(timeout=2)
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
