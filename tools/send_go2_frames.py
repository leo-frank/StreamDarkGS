#!/usr/bin/env python3
"""Read-only GO2 camera capture, local recording and conservative HTTP upload.

No automatic POST retries: the current receiver does not support idempotency.
SDK imports are lazy so --frames-dir can test the tunnel without a robot.
"""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import cv2
import numpy as np


def request(server, route, payload=None, timeout=15):
    headers = {'Content-Type': 'image/jpeg'} if route == '/frame' else {}
    req = Request(server.rstrip('/') + route, data=payload, headers=headers,
                  method='GET' if payload is None else 'POST')
    with urlopen(req, timeout=timeout) as response:
        return response.read()


def arguments():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--interface', help='Network interface connected to GO2, e.g. enp3s0')
    p.add_argument('--server', default='http://127.0.0.1:18765')
    p.add_argument('--fps', type=float, default=1)
    p.add_argument('--count', type=int, default=60)
    p.add_argument('--capture-only', action='store_true')
    p.add_argument('--frames-dir', type=Path, help='Simulation: read existing JPGs instead of the SDK')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--timeout', type=float, default=15)
    args = p.parse_args()
    if not 0 < args.fps <= 30 or args.count < 1 or args.timeout <= 0:
        p.error('fps must be in (0,30]; count and timeout must be positive')
    if not args.frames_dir and not args.interface:
        p.error('--interface is required when using the GO2 camera')
    return args


def run(args):
    if not args.capture_only:
        status = json.loads(request(args.server, '/status', timeout=args.timeout))
        if status.get('finished') or status.get('received', 0) != 0:
            raise RuntimeError('Receiver is not fresh. Start a NEW receiver session before uploading.')
    sources = None
    if args.frames_dir:
        sources = sorted(args.frames_dir.glob('*.jpg'))
        if len(sources) < args.count:
            raise ValueError(f'Need {args.count} JPGs, found {len(sources)}')
    else:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient
        ChannelFactoryInitialize(0, args.interface)
        client = VideoClient()
        client.SetTimeout(3.0)
        client.Init()

    output = args.output_dir or Path('go2_captures') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    print(f'[local-output] {output.resolve()}', flush=True)
    (output / 'session.json').write_text(json.dumps({
        'fps_target': args.fps, 'count_target': args.count, 'interface': args.interface,
        'server': args.server, 'capture_only': args.capture_only,
        'source': str(args.frames_dir) if sources is not None else 'GO2 VideoClient',
        'note': 'Capture-complete timestamps, NOT camera exposure timestamps. No automatic POST retries.'
    }, indent=2), encoding='utf-8')
    next_capture = time.monotonic()
    with (output / 'events.jsonl').open('a', encoding='utf-8') as log:
        def record(**fields):
            text = json.dumps(fields, ensure_ascii=False)
            log.write(text + '\n')
            log.flush()
            print(text, flush=True)
        try:
            for index in range(args.count):
                time.sleep(max(0, next_capture - time.monotonic()))
                started = time.monotonic()
                if sources is not None:
                    data = sources[index].read_bytes()
                else:
                    code, data = client.GetImageSample()
                    if code != 0:
                        raise RuntimeError(f'GO2 GetImageSample failed: code={code}')
                captured_at = time.time_ns()
                image = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError('Camera returned an undecodable image')
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                payload = encoded.tobytes()
                name = f'frame_{index:06d}.jpg'
                (output / name).write_bytes(payload)  # Save BEFORE upload.
                record(event='saved', index=index, file=name, captured_at_unix_ns=captured_at,
                       width=image.shape[1], height=image.shape[0],
                       capture_prepare_ms=(time.monotonic()-started)*1000)
                if not args.capture_only:
                    record(event='upload_started', index=index, file=name)
                    upload_started = time.monotonic()
                    reply = request(args.server, '/frame', payload, args.timeout).decode()
                    record(event='acknowledged', index=index, server_frame=reply,
                           upload_ms=(time.monotonic()-upload_started)*1000)
                # Never send a burst to catch up. Slow capture/network reduces effective FPS.
                next_capture = max(next_capture + 1 / args.fps, time.monotonic())
            if not args.capture_only:
                record(event='finish_started')
                request(args.server, '/finish', b'', args.timeout)
            record(event='complete', saved=args.count, acknowledged=0 if args.capture_only else args.count)
        except BaseException as exc:
            record(event='stopped', error=repr(exc),
                   note='No retry and no automatic finish on error. Saved files retained; inspect receiver before recovery.')
            raise


if __name__ == '__main__':
    try:
        run(arguments())
    except KeyboardInterrupt:
        print('Stopped locally. Receiver remains open; see README for graceful finish.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
