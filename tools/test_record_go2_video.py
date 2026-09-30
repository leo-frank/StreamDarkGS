import argparse
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from record_go2_video import record, write_concat


class RecordingTests(unittest.TestCase):
    def args(self, root):
        return argparse.Namespace(output_dir=root / 'capture', interface='mock',
                                  seconds=.12, fps=20, no_video=True)

    def test_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_concat(root, [10, 10.1, 10.4], 10.5)
            content = (root / 'frames.ffconcat').read_text()
            self.assertEqual(content.count('duration 0.100000000'), 2)
            self.assertIn('duration 0.300000000', content)
            self.assertEqual(content.count('frame_000002.jpg'), 2)

    def test_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            _, jpeg = cv2.imencode('.jpg', np.zeros((16, 24, 3), np.uint8))
            root = record(args, lambda: (0, jpeg.tobytes()))
            summary = json.loads((root / 'recording.json').read_text())
            self.assertEqual(summary['status'], 'complete')
            self.assertGreater(summary['frames'], 0)
            self.assertEqual(summary['frames'], len(list((root / 'frames').glob('*.jpg'))))
            self.assertEqual(summary['frames'], len((root / 'timestamps.jsonl').read_text().splitlines()))
            with self.assertRaises(FileExistsError):
                record(args, lambda: (0, jpeg.tobytes()))

    def test_camera_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, 'code=1'):
                record(args, lambda: (1, b''))
            summary = json.loads((args.output_dir / 'recording.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['frames'], 0)


if __name__ == '__main__':
    unittest.main()
