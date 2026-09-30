#!/usr/bin/env python3
"""Record GO2 RGB frames locally; build a variable-frame-rate MP4 using receive times.

Read-only camera access. No robot motion commands and no remote uploads.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np


def write_concat(root, stamps, end_time):
    """Timestamp-based durations; ffmpeg quantizes timestamps to its timebase."""
    if not stamps:
        return
    lines = ['ffconcat version 1.0']
    for i, stamp in enumerate(stamps):
        duration = (stamps[i+1] if i+1 < len(stamps) else end_time) - stamp
        lines += [f"file 'frames/frame_{i:06d}.jpg'", f'duration {max(duration, .001):.9f}']
    # Repeat the final image so the concat demuxer honors the final duration.
    lines.append(f"file 'frames/frame_{len(stamps)-1:06d}.jpg'")
    (root / 'frames.ffconcat').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def record(args, get_image):
    root = args.output_dir or Path('go2_videos') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'frames').mkdir()
    print(f'[output] {root.resolve()}', flush=True)
    stamps, shape = [], None
    status, error = 'complete', None
    start = time.monotonic()
    next_capture = start
    try:
        with (root / 'timestamps.jsonl').open('w', encoding='utf-8') as log:
            while time.monotonic() < start + args.seconds:
                time.sleep(max(0, next_capture-time.monotonic()))
                if time.monotonic() >= start + args.seconds:
                    break
                request_start = time.monotonic()
                code, data = get_image()
                received = time.monotonic()
                if code != 0:
                    raise RuntimeError(f'GetImageSample failed: code={code}')
                image = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError('Undecodable camera image')
                if shape is not None and image.shape != shape:
                    raise RuntimeError('Camera image dimensions changed; start a new recording')
                shape = image.shape
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                index = len(stamps)
                filename = f'frames/frame_{index:06d}.jpg'
                (root / filename).write_bytes(encoded.tobytes())
                stamps.append(received)
                info = dict(index=index, file=filename, received_elapsed_s=received-start,
                            request_ms=(received-request_start)*1000, recorded_unix_ns=time.time_ns())
                log.write(json.dumps(info)+'\n')
                log.flush()
                if index % max(round(args.fps), 1) == 0:
                    print(f'[capture] frames={index+1} elapsed={received-start:.1f}s', flush=True)
                next_capture = max(next_capture + 1/args.fps, time.monotonic())
    except KeyboardInterrupt:
        status = 'interrupted'
        print('[stop] Finishing saved frames. Do not interrupt again.', flush=True)
    except Exception as exc:
        status, error = 'failed', str(exc)
    end = time.monotonic()
    write_concat(root, stamps, end)
    summary = dict(status=status, error=error, interface=args.interface, target_fps=args.fps,
                   requested_seconds=args.seconds, elapsed_seconds=end-start, frames=len(stamps),
                   measured_fps=((len(stamps)-1)/(stamps[-1]-stamps[0])) if len(stamps)>1 else 0,
                   image_shape=shape, timestamp_note='SDK receive completion, not camera exposure; MP4 starts at first received frame.')
    (root / 'recording.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if stamps and not args.no_video:
        if shutil.which('ffmpeg'):
            result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n',
                '-f', 'concat', '-safe', '0', '-i', 'frames.ffconcat', '-vsync', 'vfr',
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-crf', '18',
                '-pix_fmt', 'yuv420p', 'camera.mp4'], cwd=root)
            if result.returncode:
                print('[warning] MP4 export failed; JPEGs and timestamps remain available.', file=sys.stderr)
        else:
            print('[warning] ffmpeg missing; convert frames.ffconcat on the GPU computer later.', file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if error or not stamps:
        raise RuntimeError(error or 'No frames recorded')
    return root


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--interface', required=True)
    p.add_argument('--fps', type=float, default=15)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--no-video', action='store_true', help='Keep JPEGs/timestamps; convert to MP4 later')
    args = p.parse_args()
    if not 0 < args.fps <= 60 or not 0 < args.seconds <= 3600:
        p.error('fps must be in (0,60]; seconds in (0,3600]')
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.video.video_client import VideoClient
    ChannelFactoryInitialize(0, args.interface)
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    record(args, client.GetImageSample)


if __name__ == '__main__':
    main()
