"""Summarize the fixed W3/S2 configuration from run_go2_receiver.sh."""
import argparse
import json
from pathlib import Path
import re

from benchmark_pipeline import summarize


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    log = (root / 'run.log').read_text(errors='replace')
    config_path = root / 'case.json'
    if not config_path.exists():
        accepted = re.findall(r'\[stream-timing\].*?accepted=(\d+)', log)
        frames = int(accepted[-1]) if accepted else 60
        config = dict(window=3, window_stride=2, create_stride=2, timing='wall', mode='stream',
                      fps=1, seconds=max(frames, 1), online_optimization=True,
                      online_steps=10, online_interval=1, online_window_multiplier=5,
                      low_latency_pipeline=True, preview_noncreation_frames=True,
                      save_viewer=(root / 'viewer_frames').exists(),
                      note='Fixed run_go2_receiver.sh settings; phase boundaries use nominal 1 FPS input index, not actual wall time.')
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    summarize(root)


if __name__ == '__main__':
    main()
