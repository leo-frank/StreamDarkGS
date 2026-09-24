#!/usr/bin/env python3
"""Window/create-stride timing experiments, with real headless viewer work."""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def integers(value):
    result = [int(v) for v in value.split(',')]
    if not result or any(v < 1 for v in result):
        raise argparse.ArgumentTypeError('Expected positive comma-separated integers')
    return result


def arguments():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--mode', choices=['matrix', 'stream', 'summarize'], default='matrix')
    p.add_argument('--windows', type=integers, default=[3, 5, 10, 30])
    p.add_argument('--create-strides', type=integers, default=[1, 2, 4])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--online-optimization', action='store_true')
    p.add_argument('--preview-after-optimization', action='store_true')
    p.add_argument('--low-latency-pipeline', action='store_true')
    p.add_argument('--preview-noncreation-frames', action='store_true')
    p.add_argument('--online-steps', type=int, default=10)
    p.add_argument('--online-interval', type=int, default=1)
    p.add_argument('--online-window-multiplier', type=float, default=5)
    p.add_argument('--timing', choices=['sync', 'wall'], default=None)
    p.add_argument('--video', default='/home/zxn/disk/Pi3-main/cvpr/our4/微信视频2025-11-08_213842_960.mp4')
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--fps', type=float, default=3)
    p.add_argument('--frames-dir', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.low_latency_pipeline and args.preview_after_optimization:
        p.error('--low-latency-pipeline conflicts with --preview-after-optimization')
    if min(args.online_steps, args.online_interval, args.online_window_multiplier) <= 0:
        p.error('online steps/interval/window multiplier must be positive')
    if min(args.seconds, args.fps, args.repeats) <= 0 or any(w < 2 for w in args.windows):
        p.error('seconds/fps/repeats must be positive; windows must be >=2')
    if args.mode == 'stream' and (len(args.windows) != 1 or len(args.create_strides) != 1 or args.repeats != 1):
        p.error('stream requires one window, one create stride, and repeats=1')
    if args.mode == 'summarize' and args.output_root is None:
        p.error('summarize requires --output-root')
    args.timing = args.timing or ('wall' if args.mode == 'stream' else 'sync')
    args.output_root = (args.output_root or REPO / 'output' / ('pipeline_timing_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if args.frames_dir:
        args.frames_dir = args.frames_dir.resolve()
    return args


def case_command(args, window, create_stride, output, frames):
    command = [sys.executable, str(REPO / 'live_gaussians_from_rgbd.py'),
        '--input_mode', 'stream' if args.mode == 'stream' else 'replay',
        '--replay_fps', '0', '--image_dir', str(frames),
        '--pi3_root', '/home/zxn/disk/Pi3-main',
        '--pi3_ckpt', '/home/zxn/.cache/huggingface/hub/models--yyfz233--Pi3/snapshots/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors',
        '--mvinverse_ckpt', '/home/zxn/.cache/huggingface/hub/models--maddog241--mvinverse/snapshots/ac2d62d9ab2d8e23370dc4de5e6543cd52662c0e',
        '--output_path', str(output / 'gaussian_map.pt'),
        '--creation_material_align_to_map', '--creation_material_align_roughness_mode', 'region_constant',
        '--creation_material_align_region_source', 'sam',
        '--creation_material_align_sam_max_color_distance', '.15',
        '--creation_material_align_global_max_log_offset', '.05',
        '--creation_material_align_cluster_max_log_offset', '.8',
        '--creation_material_align_min_valid_ratio', '.5',
        '--creation_material_align_cluster_min_pixels', '512',
        '--sam2_ckpt', '../sam2/checkpoints/sam2.1_hiera_base_plus.pt',
        '--sam2_config', 'configs/sam2.1/sam2.1_hiera_b+.yaml', '--sam2_device', 'cuda',
        '--window_size', str(window), '--window_stride', str(window - 1),
        '--fusion_frame_stride', str(create_stride), '--input_frame_stride', '1',
        '--mvinverse_overlap_policy', 'first', '--pi3_overlap_policy', 'first',
        '--no_mvinverse_window_align_to_overlap',
        '--online_global_optimization_steps', str(args.online_steps if getattr(args, 'online_optimization', False) else 0),
        '--global_optimization_steps', '0',
        '--profile_timing', args.timing]
    if getattr(args, 'online_optimization', False):
        command += ['--global_optimization', '--online_global_optimization',
                    '--online_global_optimization_interval', str(args.online_interval),
                    '--online_global_optimization_window_multiplier', str(args.online_window_multiplier)]
    if getattr(args, 'preview_after_optimization', False):
        command += ['--preview_after_online_optimization']
    if getattr(args, 'low_latency_pipeline', False):
        command += ['--low_latency_pipeline']
    if getattr(args, 'preview_noncreation_frames', False):
        command += ['--preview_noncreation_frames']
    if args.mode == 'matrix':
        command += ['--profile_viewer']
    else:
        command += ['--stream_capture_fps', str(args.fps), '--stream_port', str(args.port),
                    '--stream_queue_size', str(max(256, math.ceil(args.seconds * args.fps) + 8)),
                    '--stream_bootstrap_frames', '0', '--profile_exit_after_stream']
    return command


def percentile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    low = int(at)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (at - low)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    overview, details = [], []
    configs = [root / 'case.json'] if (root / 'case.json').exists() else sorted(root.glob('*/case.json'))
    for config_path in configs:
        config = json.loads(config_path.read_text())
        log = config_path.with_name('run.log')
        if not log.exists():
            continue
        lines = log.read_text(errors='replace').splitlines()
        groups = {}
        summary = dict(case=config_path.parent.name, window=config['window'], window_stride=config['window_stride'],
                       create_stride=config['create_stride'], timing=config['timing'], mode=config['mode'],
                       online_optimization=config.get('online_optimization', False),
                       low_latency_pipeline=config.get('low_latency_pipeline', False),
                       preview_noncreation_frames=config.get('preview_noncreation_frames', False),
                       save_viewer=config.get('save_viewer', False),
                       parallel_window_models=config.get('parallel_window_models', False),
                       online_steps=config.get('online_steps', 0),
                       online_interval=config.get('online_interval', ''),
                       online_window_multiplier=config.get('online_window_multiplier', ''))
        seen = {}
        for line in lines:
            match = re.match(r'\[([^]]+)\]\s*(.*)', line)
            if not match:
                continue
            tag, body = match.groups()
            fields = dict(re.findall(r'(\w+)=([^\s]+)', body))
            if tag == 'profile-run':
                summary.update(fields)
            if tag == 'profile-startup':
                summary.update(fields)
            if tag == 'viewer-submit':
                summary['viewer_submissions'] = summary.get('viewer_submissions', 0) + 1
                summary['viewer_dropped'] = summary.get('viewer_dropped', 0) + (fields.get('dropped_stale') == 'True')
            if tag == 'viewer-timing' and 'render_ms' in fields:
                summary['viewer_renders'] = summary.get('viewer_renders', 0) + 1
            if tag == 'frame-timing':
                summary['creation_frames'] = summary.get('creation_frames', 0) + 1
            if tag == 'window-timing':
                summary['windows_processed'] = summary.get('windows_processed', 0) + 1
            seen[tag] = seen.get(tag, 0) + 1
            partial = tag in {'profile-window', 'profile-pi3', 'profile-mvinverse', 'profile-material_window'} and int(fields.get('frames', config['window'])) != config['window']
            frame_match = re.search(r'frame_(\d+)', body)
            stream_match = re.search(r'stream_\d+_(\d+)', body)
            frame_number = int(frame_match[1]) if frame_match else (int(stream_match[1]) + 1 if stream_match else None)
            phase = None
            if frame_number is not None:
                second = (frame_number - 1) / config['fps']
                phase = ['early', 'middle', 'late'][min(int(second / (config['seconds'] / 3)), 2)]
            for key, value in fields.items():
                if not key.endswith('_ms'):
                    continue
                try:
                    number = float(value)
                except ValueError:
                    continue
                for scope in ['all'] + (['steady_full'] if seen[tag] > 1 and not partial else []) + ([phase] if phase else []):
                    groups.setdefault((tag, key, scope), []).append(number)
        summary['complete'] = 'processing_ms' in summary and any(l.startswith('[save]') for l in lines)
        process_path = config_path.with_name('process.json')
        if process_path.exists():
            summary.update(json.loads(process_path.read_text()))
            summary['complete'] = summary['complete'] and summary['returncode'] == 0
        if 'processing_ms' in summary:
            summary['processing_s'] = float(summary['processing_ms']) / 1000
            summary['effective_input_fps'] = float(summary['frames']) / max(summary['processing_s'], 1e-9)
        for (tag, metric, scope), values in sorted(groups.items()):
            details.append(dict(case=summary['case'], module=tag, metric=metric, scope=scope,
                                count=len(values), mean_ms=statistics.mean(values), p50_ms=percentile(values, .5),
                                p95_ms=percentile(values, .95), max_ms=max(values), sum_ms=sum(values)))
        overview.append(summary)
    write_csv(root / 'summary.csv', overview)
    write_csv(root / 'module_times.csv', details)
    print(f'[summary] {root / "summary.csv"}\n[modules] {root / "module_times.csv"}', flush=True)


def main():
    args = arguments()
    if args.mode == 'summarize':
        summarize(args.output_root)
        return
    frames = args.frames_dir or args.output_root / 'frames_3fps'
    cases = [(w, c, r) for w in args.windows for c in args.create_strides for r in range(1, args.repeats + 1)]
    print(f'[experiment] {args.output_root}', flush=True)
    if args.dry_run:
        for w, c, r in cases:
            out = args.output_root / f'w{w}s{w-1}_c{c}_r{r}'
            print(shlex.join(case_command(args, w, c, out, frames)))
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    if args.mode == 'matrix' and not args.frames_dir:
        frames.mkdir()
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-t', str(args.seconds),
                        '-i', args.video, '-vf', f'fps={args.fps}', '-q:v', '2', str(frames / 'frame_%06d.jpg')], check=True)
    if args.mode == 'matrix' and not list(frames.glob('*.jpg')):
        raise ValueError(f'No JPG frames in {frames}')
    failed = []
    for w, c, r in cases:
        out = args.output_root / f'w{w}s{w-1}_c{c}_r{r}'
        out.mkdir()
        command = case_command(args, w, c, out, frames)
        config = dict(window=w, window_stride=w-1, create_stride=c, repeat=r, timing=args.timing,
                      mode=args.mode, fps=args.fps, seconds=args.seconds, frames_dir=str(frames), command=command,
                      debug_images=False, offline_export=False, online_optimization=args.online_optimization,
                      preview_after_optimization=args.preview_after_optimization,
                      low_latency_pipeline=args.low_latency_pipeline,
                      preview_noncreation_frames=args.preview_noncreation_frames,
                      online_steps=args.online_steps if args.online_optimization else 0,
                      online_interval=args.online_interval,
                      online_window_multiplier=args.online_window_multiplier)
        (out / 'case.json').write_text(json.dumps(config, indent=2, ensure_ascii=False))
        print(f'[case] {out.name}: W={w} S={w-1} create={c}', flush=True)
        if args.mode == 'stream':
            print(f'[phone] http://<电脑局域网IP>:{args.port}/profile', flush=True)
        started = time.perf_counter()
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1')
        with (out / 'run.log').open('w') as log:
            with subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    raise
        (out / 'process.json').write_text(json.dumps(dict(returncode=code, wall_seconds=time.perf_counter()-started), indent=2))
        if code:
            failed.append(out.name)
        summarize(args.output_root)
    if failed:
        raise SystemExit('Failed cases: ' + ', '.join(failed))


if __name__ == '__main__':
    main()
