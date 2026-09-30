#!/usr/bin/env python3
"""Subscribe to GO2 front-camera DDS on the onboard ROS 2 computer.

Decode the firmware's H.264 packets with ffmpeg, then save and upload selected
JPEGs to the existing GO2 HTTP receiver. Run with ROS_DOMAIN_ID=0 and
CYCLONEDDS_URI selecting the robot-facing interface (eth1 at this site).
"""

import argparse
from collections import deque
import datetime
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen


def h264_packet(raw):
    if len(raw) < 24 or raw[:2] != b'\x00\x01':
        raise ValueError(f'unrecognized CDR header: {raw[:16].hex()}')
    _frame_time, resolution, size = struct.unpack_from('<QII', raw, 4)
    trailing = len(raw) - 20 - size
    if resolution != 720 or trailing < 0 or trailing > 3:
        raise ValueError(f'unrecognized video layout: resolution={resolution} size={size} bytes={len(raw)}')
    packet = raw[20:20 + size]
    if not packet.startswith((b'\x00\x00\x00\x01', b'\x00\x00\x01')):
        raise ValueError(f'expected Annex-B H.264 packet; first16={packet[:16].hex()}')
    return packet


def request(server, route, payload=None, timeout=15):
    headers = {'Content-Type': 'image/jpeg'} if route == '/frame' else {}
    req = Request(server.rstrip('/') + route, data=payload, headers=headers,
                  method='GET' if payload is None else 'POST')
    with urlopen(req, timeout=timeout) as response:
        return response.read()


class Decoder:
    def __init__(self, ffmpeg_log, on_frame=None, ffmpeg='ffmpeg', history_size=0):
        self.log = ffmpeg_log.open('wb')
        self.on_frame = on_frame
        self.reader_error = None
        self.process = subprocess.Popen([
            ffmpeg, '-hide_banner', '-loglevel', 'warning',
            '-probesize', '32', '-analyzeduration', '0',
            '-fflags', '+genpts', '-f', 'h264', '-i', 'pipe:0',
            '-vsync', '0', '-an', '-f', 'image2pipe',
            '-c:v', 'mjpeg', '-q:v', '3', 'pipe:1',
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.condition = threading.Condition()
        self.latest = None
        self.history = deque(maxlen=history_size) if history_size else None
        self.sequence = 0
        self.closed = False
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        pending = bytearray()
        try:
            while True:
                chunk = self.process.stdout.read(65536)
                if not chunk:
                    break
                pending.extend(chunk)
                while True:
                    start = pending.find(b'\xff\xd8')
                    if start < 0:
                        # SOI may be split across two pipe reads.
                        tail = pending[-1:] if pending.endswith(b'\xff') else b''
                        pending.clear()
                        pending.extend(tail)
                        break
                    if start:
                        del pending[:start]
                    end = pending.find(b'\xff\xd9', 2)
                    if end < 0:
                        if len(pending) > 8_000_000:
                            raise RuntimeError('ffmpeg JPEG output exceeded 8 MB')
                        break
                    frame = bytes(pending[:end + 2])
                    del pending[:end + 2]
                    with self.condition:
                        self.latest = (frame, time.time_ns())
                        self.sequence += 1
                        if self.history is not None:
                            self.history.append((self.sequence, self.latest))
                        self.condition.notify_all()
                    if self.on_frame is not None:
                        self.on_frame(frame)
        except Exception as exc:
            self.reader_error = exc
        finally:
            with self.condition:
                self.closed = True
                self.condition.notify_all()

    def feed(self, packet):
        if self.process.poll() is not None:
            raise RuntimeError(f'ffmpeg exited with code {self.process.returncode}')
        self.process.stdin.write(packet)

    def next_frame(self, after, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence <= after and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('No decoded camera frame; inspect ffmpeg.log')
                self.condition.wait(remaining)
            if self.sequence <= after:
                detail = f': {self.reader_error}' if self.reader_error else ''
                raise RuntimeError(f'ffmpeg stopped before producing another frame{detail}; inspect ffmpeg.log')
            return self.sequence, self.latest

    def frames_after(self, after, timeout):
        if self.history is None:
            raise RuntimeError('Decoder frame history is disabled')
        self.next_frame(after, timeout)
        with self.condition:
            return self.sequence, [item for item in self.history if item[0] > after]

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        self.log.close()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='http://127.0.0.1:18766')
    parser.add_argument('--fps', type=float, default=1)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=15)
    args = parser.parse_args()
    if not 0 < args.fps <= 1 or args.count < 1 or args.timeout <= 0:
        parser.error('fps must be in (0,1]; count and timeout must be positive')
    if shutil.which('ffmpeg') is None:
        parser.error('ffmpeg is required on the onboard computer to decode H.264')
    return args


def run(args):
    if not args.capture_only:
        status = json.loads(request(args.server, '/status', timeout=args.timeout))
        if status.get('finished') or status.get('received', 0) != 0:
            raise RuntimeError('Receiver is not fresh. Start a NEW receiver session.')

    # Imports are lazy so --help and unit tests work outside the onboard ROS env.
    import rclpy
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy
    from unitree_go.msg import Go2FrontVideoData

    output = args.output_dir or Path('go2_captures') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    print(f'[local-output] {output.resolve()}', flush=True)
    (output / 'session.json').write_text(json.dumps({
        'source': 'ROS2 /frontvideostream, raw H.264, onboard eth1',
        'fps_target': args.fps, 'count_target': args.count,
        'server': args.server, 'capture_only': args.capture_only,
        'note': 'Selected JPEG timestamp is decode completion, not camera exposure. No automatic POST retries.',
    }, indent=2), encoding='utf-8')
    decoder = Decoder(output / 'ffmpeg.log')
    rclpy.init()
    node = rclpy.create_node('go2_remote_camera_sender')
    counts = {'packets': 0, 'errors': 0}

    def on_video(raw):
        try:
            decoder.feed(h264_packet(raw))
            counts['packets'] += 1
        except (ValueError, BrokenPipeError, RuntimeError) as exc:
            counts['errors'] += 1
            if counts['errors'] <= 3:
                print(f'[camera-warning] {exc}', file=sys.stderr, flush=True)

    qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
    subscription = node.create_subscription(Go2FrontVideoData, '/frontvideostream', on_video, qos, raw=True)
    spin_error = []

    def spin():
        try:
            rclpy.spin(node)
        except Exception as exc:
            spin_error.append(exc)

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()
    try:
        with (output / 'events.jsonl').open('w', encoding='utf-8') as event_log:
            def record(**fields):
                line = json.dumps(fields, ensure_ascii=False)
                event_log.write(line + '\n')
                event_log.flush()
                print(line, flush=True)

            next_capture = time.monotonic()
            for index in range(args.count):
                time.sleep(max(0, next_capture - time.monotonic()))
                if spin_error:
                    raise RuntimeError(f'ROS2 subscription stopped: {spin_error[0]}')
                with decoder.condition:
                    sequence = decoder.sequence
                sequence, (jpeg, decoded_at_ns) = decoder.next_frame(sequence, args.timeout)
                name = f'frame_{index:06d}.jpg'
                (output / name).write_bytes(jpeg)
                record(event='saved', index=index, file=name, decoded_at_unix_ns=decoded_at_ns,
                       jpeg_bytes=len(jpeg), ros_packets=counts['packets'])
                if not args.capture_only:
                    record(event='upload_started', index=index, file=name)
                    started = time.monotonic()
                    reply = request(args.server, '/frame', jpeg, args.timeout).decode()
                    record(event='acknowledged', index=index, server_frame=reply,
                           upload_ms=(time.monotonic() - started) * 1000)
                next_capture = time.monotonic() + 1 / args.fps
            if not args.capture_only:
                record(event='finish_started')
                request(args.server, '/finish', b'', args.timeout)
            record(event='complete', saved=args.count,
                   acknowledged=0 if args.capture_only else args.count,
                   ros_packets=counts['packets'], camera_warnings=counts['errors'])
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=2)
        node.destroy_subscription(subscription)
        node.destroy_node()
        decoder.close()


if __name__ == '__main__':
    try:
        run(arguments())
    except KeyboardInterrupt:
        print('Stopped locally. Receiver remains open; use /finish only after checking /status.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
