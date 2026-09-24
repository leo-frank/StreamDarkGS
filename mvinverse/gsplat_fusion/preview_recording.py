"""Persist published JPEG bytes without an additional render or re-encoding."""
import json
from pathlib import Path
import time


def save_preview(directory, frame, version, previews, metadata):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    stem = f'{stamp}_{version}_{Path(frame).name}'
    files = {}
    for channel in ('albedo', 'relit'):
        path = root / channel / f'{stem}.jpg'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(previews[channel])
        files[channel] = str(path.relative_to(root))
    with (root / 'frames.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(frame=frame, version=version, saved_unix_ns=stamp,
                                     files=files, metadata=metadata), ensure_ascii=False) + '\n')
