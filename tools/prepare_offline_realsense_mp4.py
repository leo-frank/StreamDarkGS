#!/usr/bin/env python3
"""Select one clear frame per video second and write Pi3X camera sidecars."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--camera-info', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start-sec', type=float, default=0)
    parser.add_argument('--duration-sec', type=float, default=60,
                        help='0 uses the rest of the video')
    parser.add_argument('--output-fps', type=float, default=1)
    args = parser.parse_args()
    if args.start_sec < 0 or args.duration_sec < 0 or args.output_fps <= 0:
        parser.error('start-sec and duration-sec must be nonnegative; output-fps positive')
    return args


def calibration(path, width, height):
    source = json.loads(path.read_text(encoding='utf-8'))
    if (int(source['width']), int(source['height'])) != (width, height):
        raise ValueError(f'video is {width}x{height}, but camera calibration is '
                         f'{source["width"]}x{source["height"]}; do not reuse K after a crop or resize')
    k = np.asarray(source['K'], dtype=np.float64).reshape(3, 3)
    d = np.asarray(source.get('D', []), dtype=np.float64)
    if (not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or not np.isfinite(d).all()
            or np.any(np.abs(d) > 1e-8)):
        raise ValueError('invalid or distorted color calibration; Pi3X sidecar needs undistorted images')
    return {
        'width': width, 'height': height,
        'fx': float(k[0, 0]), 'fy': float(k[1, 1]),
        'cx': float(k[0, 2]), 'cy': float(k[1, 2]),
        'distortion_model': source.get('distortion_model', 'none'),
        'distortion_coefficients': d.tolist(),
    }


def quality_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 480:
        height = round(gray.shape[0] * 480 / gray.shape[1])
        gray = cv2.resize(gray, (480, height), interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    return float(cv2.Laplacian(smooth, cv2.CV_32F).var())


def run(args):
    if not args.video.is_file() or not args.camera_info.is_file():
        raise FileNotFoundError('video or color camera calibration is missing')
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open video: {args.video}')
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(video_fps) or video_fps <= 0 or total_frames < 1:
            raise RuntimeError('video has no valid frame rate or frame count')
        if args.output_fps > video_fps:
            raise ValueError('output-fps cannot exceed video fps')
        intrinsics = calibration(args.camera_info, width, height)
        first_frame = round(args.start_sec * video_fps)
        last_frame = min(total_frames, round((args.start_sec + args.duration_sec) * video_fps)) \
            if args.duration_sec else total_frames
        if first_frame >= last_frame:
            raise ValueError('selected time range contains no video frames')
        if args.output_dir.exists():
            raise FileExistsError(f'output directory already exists: {args.output_dir}')
        args.output_dir.mkdir(parents=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        print(f'[video] {args.video} {width}x{height} fps={video_fps:.6f} '
              f'frames={total_frames} duration_s={total_frames / video_fps:.3f}', flush=True)
        print(f'[calibration] fx={intrinsics["fx"]:.6f} fy={intrinsics["fy"]:.6f} '
              f'cx={intrinsics["cx"]:.6f} cy={intrinsics["cy"]:.6f}', flush=True)
        print(f'[selection] source_frames={first_frame}..{last_frame - 1} '
              f'output_fps={args.output_fps:g} method=highest_denoised_laplacian', flush=True)
        manifest_path = args.output_dir.parent / 'selection.jsonl'
        selected = 0
        observed = 0
        current_slot = None
        best = None
        candidates = 0

        def save_best(manifest):
            nonlocal selected
            if best is None:
                return
            name = f'frame_{current_slot:06d}'
            image_path = args.output_dir / f'{name}.png'
            if not cv2.imwrite(str(image_path), best['image']):
                raise OSError(f'cannot write {image_path}')
            metadata = {
                'source': 'offline RealSense color MP4',
                'color_intrinsics': intrinsics,
                'source_video': str(args.video.resolve()),
                'source_frame_index': best['frame_index'],
                'source_timestamp_ms': best['frame_index'] / video_fps * 1000,
                'video_fps': video_fps,
                'selection_score': best['score'],
                'candidates_in_interval': candidates,
            }
            (args.output_dir / f'{name}.json').write_text(
                json.dumps(metadata, indent=2), encoding='utf-8')
            manifest.write(json.dumps({
                'output': image_path.name, 'slot': current_slot,
                'source_frame_index': best['frame_index'],
                'source_timestamp_ms': metadata['source_timestamp_ms'],
                'score': best['score'], 'candidates': candidates,
            }) + '\n')
            selected += 1

        with manifest_path.open('w', encoding='utf-8') as manifest:
            for frame_index in range(first_frame, last_frame):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'video ended while decoding frame {frame_index}')
                if frame.shape[:2] != (height, width):
                    raise ValueError(f'frame {frame_index} changed dimensions; camera K is not valid')
                slot = int((frame_index - first_frame) * args.output_fps / video_fps + 1e-9)
                if current_slot is not None and slot != current_slot:
                    save_best(manifest)
                    best = None
                    candidates = 0
                current_slot = slot
                score = quality_score(frame)
                candidates += 1
                observed += 1
                if best is None or score > best['score']:
                    best = {'image': frame, 'frame_index': frame_index, 'score': score}
            save_best(manifest)
        print(f'[prepared] selected={selected} decoded={observed} '
              f'input_dir={args.output_dir} manifest={manifest_path}', flush=True)
        return selected
    finally:
        cap.release()


if __name__ == '__main__':
    if run(arguments()) < 3:
        raise SystemExit('fewer than 3 selected frames; Pi3X window_size=3 cannot run')
