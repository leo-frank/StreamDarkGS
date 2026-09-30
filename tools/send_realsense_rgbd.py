#!/usr/bin/env python3
"""Fetch D435i frames through SSH tunnel, archive RGB-D, upload color to reconstruction."""

import argparse
import base64
from collections import deque
import datetime
import json
from pathlib import Path
import socket
import statistics
import struct
import sys
import threading
import time
from urllib.request import Request, urlopen

from send_go2_ros_frames import request


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-port', type=int, default=18800)
    parser.add_argument('--server', default='http://127.0.0.1:8765')
    parser.add_argument('--fps', type=float, default=1.0)
    parser.add_argument('--count', type=int, default=60)
    parser.add_argument('--selection', choices=('latest', 'sharpest', 'motion'), default='latest',
                        help='Select latest, highest sharpness, or lowest estimated motion blur')
    parser.add_argument('--candidate-fps', type=float, default=15.0,
                        help='Maximum camera requests per second with sharpest or motion selection')
    parser.add_argument('--selection-window-ms', type=float,
                        help='Candidate interval in ms; motion defaults to the full output interval')
    parser.add_argument('--min-sharpness', type=float, default=20.0,
                        help='Minimum denoised Laplacian score for upload in sharpest mode')
    parser.add_argument('--max-motion-blur-px', type=float, default=3.0,
                        help='Maximum estimated full-resolution exposure smear in motion mode')
    parser.add_argument('--motion-sharpness-ratio', type=float, default=0.7,
                        help='For measured high motion, reject only if sharpness is below this fraction of recent clear frames')
    parser.add_argument('--motion-fallback-ratio', type=float, default=0.4,
                        help='Reject untracked frames only if sharpness falls below this fraction of recent clear frames')
    parser.add_argument('--min-brightness', type=float, default=None,
                        help='Legacy option, ignored; brightness never filters frames')
    parser.add_argument('--timeout', type=float, default=20.0)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--capture-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.fps <= 6 or args.count < 1 or args.timeout <= 0:
        parser.error('fps must be in (0,6]; count and timeout must be positive')
    if not args.fps <= args.candidate_fps <= 30:
        parser.error('candidate-fps must be between fps and 30')
    if args.selection_window_ms is None:
        args.selection_window_ms = (1000 / args.fps if args.selection == 'motion'
                                    else min(250, 1000 / args.fps))
    if not 0 < args.selection_window_ms <= 1000 / args.fps:
        parser.error('selection-window-ms must be positive and no longer than output interval')
    if (args.min_sharpness < 0 or args.max_motion_blur_px <= 0
            or not 0 < args.motion_fallback_ratio < args.motion_sharpness_ratio < 1):
        parser.error('min-sharpness must be nonnegative; max-motion-blur-px positive; '
                     '0 < motion-fallback-ratio < motion-sharpness-ratio < 1')
    return args


def recv_exact(connection, size):
    payload = bytearray()
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise ConnectionError(f'camera stream closed after {len(payload)}/{size} bytes')
        payload.extend(chunk)
    return bytes(payload)


def read_frame(connection):
    connection.sendall(b'F')
    info_size, jpeg_size, depth_size = struct.unpack('!III', recv_exact(connection, 12))
    if not 0 < info_size <= 16_384 or not 0 < jpeg_size <= 8_000_000 or depth_size > 8_000_000:
        raise ValueError(f'invalid frame sizes: {info_size}, {jpeg_size}, {depth_size}')
    info = json.loads(recv_exact(connection, info_size))
    jpeg = recv_exact(connection, jpeg_size)
    depth_png = recv_exact(connection, depth_size)
    if not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9'):
        raise ValueError('invalid JPEG data')
    if depth_png and not depth_png.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('invalid depth PNG data')
    return info, jpeg, depth_png


def decode_quality(jpeg):
    import cv2
    import numpy as np

    # JPEG decoder downsamples before materializing pixels; this is cheaper
    # than decoding the full 1280x720 image only to resize it for scoring.
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                        cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if image is None:
        raise ValueError('Cannot decode RealSense JPEG for quality scoring')
    if image.shape[1] > 480:
        height = max(1, round(image.shape[0] * 480 / image.shape[1]))
        image = cv2.resize(image, (480, height), interpolation=cv2.INTER_AREA)
    brightness = float(image.mean())
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    sharpness = float(cv2.Laplacian(denoised, cv2.CV_32F).var())
    return image, sharpness, brightness


def quality_score(jpeg):
    _, sharpness, brightness = decode_quality(jpeg)
    return sharpness, brightness


def estimate_motion_blur(previous_image, image, previous_info, info):
    """Estimate exposure smear from robust scene motion between color frames.

    Returns (full-resolution pixels, inlier count), or (None, 0) when motion
    cannot be estimated reliably. Unknown estimates are never used to reject.
    """
    import cv2
    import numpy as np

    if previous_image is None or previous_image.shape != image.shape:
        return None, 0
    before = previous_info.get('color_timestamp_ms')
    after = info.get('color_timestamp_ms')
    exposure_ms = info.get('color_actual_exposure_ms_estimate')
    if before is None or after is None or exposure_ms is None:
        return None, 0
    delta_ms = after - before
    if not 0 < delta_ms <= 500 or not 0 < exposure_ms <= delta_ms:
        return None, 0
    corners = cv2.goodFeaturesToTrack(previous_image, maxCorners=160,
                                      qualityLevel=0.02, minDistance=8)
    if corners is None or len(corners) < 16:
        return None, 0
    tracked, valid, _ = cv2.calcOpticalFlowPyrLK(
        previous_image, image, corners, None, winSize=(15, 15), maxLevel=2)
    if tracked is None or valid is None:
        return None, 0
    valid = valid.ravel() == 1
    source = corners[valid].reshape(-1, 2)
    target = tracked[valid].reshape(-1, 2)
    if len(source) < 16:
        return None, 0
    affine, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0,
        maxIters=100, confidence=0.95)
    if affine is None or inliers is None or int(inliers.sum()) < 12:
        return None, 0
    source = source[inliers.ravel() == 1]
    predicted = source @ affine[:, :2].T + affine[:, 2]
    displacement = np.linalg.norm(predicted - source, axis=1)
    color_width = info.get('color_intrinsics', {}).get('width', image.shape[1])
    scale = color_width / image.shape[1]
    smear_px = float(np.percentile(displacement, 75) * scale * exposure_ms / delta_ms)
    return smear_px, len(source)


def upload_frame(server, jpeg, info, timeout):
    encoded = base64.b64encode(json.dumps(info).encode('utf-8')).decode('ascii')
    frame_request = Request(
        server.rstrip('/') + '/frame', data=jpeg,
        headers={'Content-Type': 'image/jpeg', 'X-Frame-Metadata': encoded}, method='POST')
    with urlopen(frame_request, timeout=timeout) as response:
        return response.read().decode()


def save_frame(directory, stem, info, jpeg, depth_png):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{stem}.jpg').write_bytes(jpeg)
    (directory / f'{stem}.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    if depth_png:
        (directory / f'{stem}_depth.png').write_bytes(depth_png)


class CandidateSampler:
    """Continuously fetch and score frames without delaying output decisions."""

    def __init__(self, connection, args):
        self.connection = connection
        self.args = args
        self.frames = deque(maxlen=max(8, int(args.candidate_fps * 2)))
        self.lock = threading.Condition()
        self.stopping = threading.Event()
        self.error = None
        self.received = 0
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, name='realsense-sampler', daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        import cv2

        # One scoring worker needs no OpenCV worker pool; avoid thread startup
        # overhead and contention with the reconstruction process.
        cv2.setNumThreads(1)
        next_request = time.monotonic()
        last_frame_number = None
        previous_image = previous_info = None
        recent_clear_scores = deque()
        try:
            while not self.stopping.is_set():
                if self.stopping.wait(max(0, next_request - time.monotonic())):
                    break
                read_start = time.perf_counter()
                info, jpeg, depth_png = read_frame(self.connection)
                read_ms = (time.perf_counter() - read_start) * 1000
                next_request = max(next_request + 1 / self.args.candidate_fps,
                                   time.monotonic())
                frame_number = info['color_frame_number']
                if frame_number == last_frame_number:
                    continue
                last_frame_number = frame_number
                score_start = time.perf_counter()
                image, sharpness, brightness = decode_quality(jpeg)
                score_ms = (time.perf_counter() - score_start) * 1000
                motion_blur_px = None
                motion_features = 0
                motion_ms = 0.0
                if self.args.selection == 'motion':
                    motion_start = time.perf_counter()
                    motion_blur_px, motion_features = estimate_motion_blur(
                        previous_image, image, previous_info, info)
                    motion_ms = (time.perf_counter() - motion_start) * 1000
                previous_image, previous_info = image, info
                received_at = time.monotonic()
                while recent_clear_scores and received_at - recent_clear_scores[0][0] > 1.5:
                    recent_clear_scores.popleft()
                reference_sharpness = (statistics.median(score for _, score in recent_clear_scores)
                                       if len(recent_clear_scores) >= 3 else None)
                sharpness_ratio = (sharpness / reference_sharpness
                                   if reference_sharpness and reference_sharpness > 0 else None)
                fallback_blurry = (self.args.selection == 'motion'
                                   and motion_blur_px is None
                                   and sharpness_ratio is not None
                                   and sharpness_ratio < self.args.motion_fallback_ratio)
                if (self.args.selection == 'motion' and motion_blur_px is not None
                        and motion_blur_px <= self.args.max_motion_blur_px):
                    recent_clear_scores.append((received_at, sharpness))
                candidate = {'info': info, 'jpeg': jpeg, 'depth_png': depth_png,
                             'sharpness': sharpness, 'brightness': brightness,
                             'motion_blur_px': motion_blur_px,
                             'motion_features': motion_features,
                             'motion_ms': motion_ms,
                             'motion_reference_sharpness': reference_sharpness,
                             'motion_sharpness_ratio': sharpness_ratio,
                             'motion_fallback_blurry': fallback_blurry,
                             'read_ms': read_ms, 'score_ms': score_ms,
                             'received_at': received_at}
                with self.lock:
                    self.frames.append(candidate)
                    self.received += 1
                    self.lock.notify_all()
        except Exception as exc:
            if not self.stopping.is_set():
                with self.lock:
                    self.error = exc
                    self.lock.notify_all()

    def wait_for_first(self, timeout):
        with self.lock:
            ready = self.lock.wait_for(lambda: self.frames or self.error, timeout=timeout)
            if self.error:
                raise self.error
            if not ready:
                raise TimeoutError('No RealSense candidate arrived before timeout')

    def recent(self, window_ms):
        now = time.monotonic()
        with self.lock:
            if self.error:
                raise self.error
            frames = [frame for frame in self.frames
                      if now - frame['received_at'] <= window_ms / 1000]
            return now, frames, self.received

    def stop(self):
        self.stopping.set()
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.thread.join(timeout=2)


def select_recent(sampler, args):
    decision_start = time.perf_counter()
    now, candidates, received = sampler.recent(args.selection_window_ms)
    if args.selection == 'motion':
        known_clear = [frame for frame in candidates
                       if frame['motion_blur_px'] is not None
                       and frame['motion_blur_px'] <= args.max_motion_blur_px]
        moving_but_detailed = [frame for frame in candidates
                               if frame['motion_blur_px'] is not None
                               and frame['motion_blur_px'] > args.max_motion_blur_px
                               and (frame['motion_sharpness_ratio'] is None
                                    or frame['motion_sharpness_ratio'] >= args.motion_sharpness_ratio)]
        unknown = [frame for frame in candidates
                   if frame['motion_blur_px'] is None
                   and not frame['motion_fallback_blurry']]
        # Keep the sharpest acceptable image in each output interval. A frame
        # with unmeasurable motion remains eligible unless the relative-detail
        # fallback identifies an abrupt loss of detail.
        eligible = known_clear + moving_but_detailed + unknown
        best = max(eligible, key=lambda frame: (frame['sharpness'], frame['received_at']),
                   default=None)
        observed_best = max(candidates, key=lambda frame: frame['sharpness'], default=None)
    else:
        eligible = [frame for frame in candidates
                    if frame['sharpness'] >= args.min_sharpness]
        best = max(eligible, key=lambda frame: frame['sharpness'], default=None)
        observed_best = max(candidates, key=lambda frame: frame['sharpness'], default=None)
    decision_ms = (time.perf_counter() - decision_start) * 1000
    return now, candidates, received, best, observed_best, decision_ms


def run(args):
    if args.min_brightness is not None:
        print('[realsense] --min-brightness is ignored; brightness does not filter frames',
              file=sys.stderr, flush=True)
    if not args.capture_only:
        status = json.loads(request(args.server, '/status', timeout=args.timeout))
        if status.get('finished') or status.get('received', 0) != 0:
            raise RuntimeError('Receiver is not fresh. Start a NEW receiver session.')
    output = args.output_dir or Path('realsense_captures') / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    print(f'[local-output] {output.resolve()}', flush=True)
    session = {'source': 'Intel RealSense D435i color with optional aligned depth via SSH tunnel',
               'fps_target': args.fps, 'count_target': args.count,
               'capture_only': args.capture_only, 'server': args.server,
               'selection': args.selection, 'candidate_fps_target': args.candidate_fps,
               'selection_window_ms': args.selection_window_ms,
               'min_sharpness': args.min_sharpness,
               'max_motion_blur_px': args.max_motion_blur_px,
               'motion_sharpness_ratio': args.motion_sharpness_ratio,
               'motion_fallback_ratio': args.motion_fallback_ratio,
               'brightness_filter_enabled': False}
    (output / 'session.json').write_text(json.dumps(session, indent=2), encoding='utf-8')
    with socket.create_connection(('127.0.0.1', args.camera_port), timeout=args.timeout) as connection:
        connection.settimeout(args.timeout)
        sampler = CandidateSampler(connection, args) if args.selection != 'latest' else None
        if sampler:
            sampler.start()
        saved = 0
        skipped = 0
        try:
            if sampler:
                sampler.wait_for_first(args.timeout)
                next_capture = time.monotonic() + args.selection_window_ms / 1000
            else:
                next_capture = time.monotonic()
            with (output / 'events.jsonl').open('w', encoding='utf-8') as log:
                for index in range(args.count):
                    time.sleep(max(0, next_capture - time.monotonic()))
                    if sampler:
                        now, candidates, received, best, observed_best, decision_ms = select_recent(sampler, args)
                        observed = len(candidates)
                        next_capture = max(next_capture + 1 / args.fps, time.monotonic())
                        if best is None:
                            skipped += 1
                            if observed_best is not None:
                                save_frame(output / 'rejected', f'window_{index:06d}',
                                           observed_best['info'], observed_best['jpeg'],
                                           observed_best['depth_png'])
                            event = {'event': 'skipped', 'index': index, 'candidates': observed,
                                     'candidate_frame_numbers': [frame['info']['color_frame_number']
                                                                 for frame in candidates],
                                     'received_total': received, 'decision_ms': round(decision_ms, 3),
                                     'best_sharpness': None if observed_best is None else observed_best['sharpness'],
                                     'best_brightness': None if observed_best is None else observed_best['brightness'],
                                     'best_motion_blur_px': None if observed_best is None else observed_best['motion_blur_px'],
                                     'best_motion_sharpness_ratio': None if observed_best is None else observed_best['motion_sharpness_ratio'],
                                     'best_motion_fallback_blurry': None if observed_best is None else observed_best['motion_fallback_blurry']}
                            log.write(json.dumps(event) + '\n')
                            log.flush()
                            print(f'[realsense] {event}', flush=True)
                            continue
                        selected_age_ms = (now - best['received_at']) * 1000
                        info, jpeg, depth_png = best['info'], best['jpeg'], best['depth_png']
                        score, brightness = best['sharpness'], best['brightness']
                    else:
                        info, jpeg, depth_png = read_frame(connection)
                        score = brightness = decision_ms = selected_age_ms = None
                        observed = 1
                        next_capture = max(next_capture + 1 / args.fps, time.monotonic())
                    name = f'frame_{index:06d}'
                    save_frame(output, name, info, jpeg, depth_png)
                    saved += 1
                    event = {'event': 'saved', 'index': index,
                             'color_frame_number': info['color_frame_number'],
                             'color_intrinsics': info['color_intrinsics'], 'depth_bytes': len(depth_png),
                             'candidates': observed, 'sharpness': score, 'brightness': brightness}
                    if sampler:
                        event.update({'candidate_frame_numbers': [frame['info']['color_frame_number']
                                                                  for frame in candidates],
                                      'selected_age_ms': round(selected_age_ms, 2),
                                      'decision_ms': round(decision_ms, 3),
                                      'camera_read_ms': round(best['read_ms'], 2),
                                      'score_ms': round(best['score_ms'], 2),
                                      'motion_ms': round(best['motion_ms'], 2),
                                      'motion_blur_px': best['motion_blur_px'],
                                      'motion_features': best['motion_features'],
                                      'motion_sharpness_ratio': best['motion_sharpness_ratio'],
                                      'motion_fallback_blurry': best['motion_fallback_blurry']})
                    if not args.capture_only:
                        upload_start = time.perf_counter()
                        event['server_frame'] = upload_frame(args.server, jpeg, info, args.timeout)
                        event['upload_ms'] = round((time.perf_counter() - upload_start) * 1000, 2)
                    log.write(json.dumps(event) + '\n')
                    log.flush()
                    print(f'[realsense] {event}', flush=True)
                if not args.capture_only and saved:
                    request(args.server, '/finish', b'', args.timeout)
        finally:
            if sampler:
                sampler.stop()
    print(f'[done] saved={saved} skipped={skipped} windows={args.count} output={output}', flush=True)
    if sampler:
        elapsed = time.monotonic() - sampler.started
        print(f'[timing] candidates={sampler.received} elapsed_s={elapsed:.2f} '
              f'actual_candidate_fps={sampler.received / elapsed:.2f}', flush=True)


if __name__ == '__main__':
    try:
        run(arguments())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
