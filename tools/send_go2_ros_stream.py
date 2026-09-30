#!/usr/bin/env python3
"""Stream GO2 H.264 over SSH, save it, and send current JPEG frames to reconstruction.

The onboard serve_go2_ros_h264.py and local SSH tunnel must already be running.
The camera is subscribed to continuously; this script never calls GetImageSample.
"""

import argparse
import cv2
import datetime
import json
import numpy as np
from pathlib import Path
import shutil
import socket
import sys
import threading
import time

from send_go2_ros_frames import Decoder, request


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-port', type=int, default=18800)
    parser.add_argument('--server', default='http://127.0.0.1:8765')
    parser.add_argument('--fps', type=float, default=1)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--selection', choices=('sharpest', 'latest'), default='sharpest',
                        help='Select the sharpest recent decoded frame (default), or the latest frame')
    parser.add_argument('--sharpness-window-ms', type=float, default=200,
                        help='Look back this many milliseconds when selecting the sharpest frame (default: 200)')
    parser.add_argument('--timeout', type=float, default=15,
                        help='Maximum wait for a new decoded frame or HTTP request')
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'),
                        help='Path to ffmpeg; required to decode the H.264 stream')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not 1 <= args.camera_port <= 65535:
        parser.error('camera-port must be in 1..65535')
    if not 0 < args.fps <= 30 or args.count < 1 or args.timeout <= 0 or args.sharpness_window_ms <= 0:
        parser.error('fps must be in (0,30]; count and timeout must be positive')
    if not args.ffmpeg:
        parser.error('ffmpeg is required; pass --ffmpeg /absolute/path/to/ffmpeg')
    return args


def sharpness_score(jpeg):
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('Cannot decode camera JPEG for sharpness scoring')
    if image.shape[1] > 480:
        height = max(1, round(image.shape[0] * 480 / image.shape[1]))
        image = cv2.resize(image, (480, height), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(image, cv2.CV_32F).var())


def run(args):
    status = json.loads(request(args.server, '/status', timeout=args.timeout))
    if status.get('finished') or status.get('received', 0) != 0:
        raise RuntimeError('Receiver is not fresh. Start a NEW receiver session.')
    try:
        camera = socket.create_connection(('127.0.0.1', args.camera_port), timeout=5)
    except OSError as exc:
        raise RuntimeError(
            f'Cannot connect to camera on port {args.camera_port}; '
            'start the onboard ROS 2 source and local SSH tunnel first'
        ) from exc

    output = args.output_dir or Path('go2_captures_ros') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    print(f'[local-output] {output.resolve()}', flush=True)
    (output / 'session.json').write_text(json.dumps({
        'source': 'GO2 ROS 2 /frontvideostream H.264 via SSH tunnel',
        'fps_target': args.fps, 'count_target': args.count,
        'server': args.server, 'camera_port': args.camera_port,
        'timestamp_note': 'JPEG decode completion, not camera exposure',
        'selection': args.selection,
        'sharpness_window_ms': args.sharpness_window_ms,
        'selection_note': 'Best Laplacian-variance score among already-decoded recent frames; all video remains in camera.h264',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    decoder = Decoder(output / 'ffmpeg.log', ffmpeg=args.ffmpeg, history_size=64)
    stopping = threading.Event()
    source_error = []
    source_stats = {'raw_bytes': 0}
    camera.settimeout(1)

    def receive_stream():
        try:
            with camera, (output / 'camera.h264').open('wb') as raw_file:
                while not stopping.is_set():
                    try:
                        chunk = camera.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        raise ConnectionError('camera stream closed')
                    raw_file.write(chunk)
                    source_stats['raw_bytes'] += len(chunk)
                    decoder.feed(chunk)
        except OSError as exc:
            if not stopping.is_set():
                source_error.append(exc)
        except Exception as exc:
            source_error.append(exc)

    receiver = threading.Thread(target=receive_stream, daemon=True)
    receiver.start()
    last_sequence = 0
    next_capture = time.monotonic()
    saved = 0
    error = None
    try:
        with (output / 'events.jsonl').open('w', encoding='utf-8') as event_log:
            def record(**fields):
                line = json.dumps(fields, ensure_ascii=False)
                event_log.write(line + '\n')
                event_log.flush()
                print(line, flush=True)

            for index in range(args.count):
                time.sleep(max(0, next_capture - time.monotonic()))
                deadline = time.monotonic() + args.timeout
                while True:
                    if source_error:
                        raise RuntimeError(f'camera stream failed: {source_error[0]}')
                    if decoder.reader_error is not None:
                        raise RuntimeError(f'ffmpeg decoder failed: {decoder.reader_error}')
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            'No fresh decoded frame; check that onboard [source] video720p '
                            'keeps increasing, then inspect eth1, DDS and ffmpeg.log'
                        )
                    try:
                        if args.selection == 'sharpest':
                            newest_sequence, candidates = decoder.frames_after(
                                last_sequence, min(remaining, 1.0))
                            last_sequence = newest_sequence
                            cutoff_ns = time.time_ns() - int(args.sharpness_window_ms * 1_000_000)
                            candidates = [item for item in candidates if item[1][1] >= cutoff_ns]
                            if not candidates:
                                continue
                            scored = [(sharpness_score(frame), sequence, frame, decoded_at_ns)
                                      for sequence, (frame, decoded_at_ns) in candidates]
                            score, sequence, jpeg, decoded_at_ns = max(scored,
                                key=lambda item: (item[0], item[1]))
                            candidate_count = len(scored)
                        else:
                            sequence, (jpeg, decoded_at_ns) = decoder.next_frame(
                                last_sequence, min(remaining, 1.0))
                            score = None
                            candidate_count = 1
                            last_sequence = sequence
                        break
                    except TimeoutError:
                        continue
                filename = f'frame_{index:06d}.jpg'
                (output / filename).write_bytes(jpeg)
                saved += 1
                record(event='saved', index=index, file=filename,
                       source_sequence=sequence, decoded_at_unix_ns=decoded_at_ns,
                       jpeg_bytes=len(jpeg), selection=args.selection,
                       candidate_count=candidate_count, sharpness_score=score,
                       selected_age_ms=(time.time_ns() - decoded_at_ns) / 1_000_000)
                record(event='upload_started', index=index, file=filename)
                started = time.monotonic()
                reply = request(args.server, '/frame', jpeg, args.timeout).decode()
                record(event='acknowledged', index=index, server_frame=reply,
                       upload_ms=(time.monotonic() - started) * 1000)
                next_capture = max(next_capture + 1 / args.fps, time.monotonic())
            record(event='finish_started')
            request(args.server, '/finish', b'', args.timeout)
            record(event='complete', saved=saved, raw_h264_bytes=source_stats['raw_bytes'])
    except BaseException as exc:
        error = repr(exc)
        with (output / 'events.jsonl').open('a', encoding='utf-8') as event_log:
            event_log.write(json.dumps({'event': 'stopped', 'error': error}, ensure_ascii=False) + '\n')
        raise
    finally:
        stopping.set()
        camera.close()
        receiver.join(timeout=2)
        decoder.close()
        (output / 'summary.json').write_text(json.dumps({
            'status': 'complete' if error is None else 'failed',
            'error': error, 'saved': saved,
            'raw_h264_bytes': source_stats['raw_bytes'],
        }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    try:
        run(arguments())
    except KeyboardInterrupt:
        print('Stopped locally. Receiver remains open; inspect /status before recovery.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
