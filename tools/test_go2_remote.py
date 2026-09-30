import argparse
import ast
import json
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tools import send_go2_frames
from tools.benchmark_pipeline import summarize
from mvinverse.gsplat_fusion.preview_recording import save_preview


class Go2RemoteTests(unittest.TestCase):
    def test_published_preview_bytes_saved_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            previews = dict(albedo=b'albedo-jpeg', relit=b'relit-jpeg')
            for version in (1, 2):
                save_preview(directory, 'frame_000001', version, previews, {'received_at': 1.0})
            root = Path(directory)
            records = [json.loads(line) for line in (root / 'frames.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0]['files'], records[1]['files'])
            for record in records:
                for channel, data in previews.items():
                    self.assertEqual((root / record['files'][channel]).read_bytes(), data)

    def test_single_go2_run_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'case.json').write_text(json.dumps(dict(window=3, window_stride=2,
                create_stride=2, timing='wall', mode='stream', fps=1, seconds=60)))
            (root / 'run.log').write_text('[viewer-save] frame=x save_ms=2.0\n'
                '[profile-run] processing_ms=100 frames=1\n[save] map.pt\n')
            summarize(root)
            self.assertTrue((root / 'summary.csv').exists())
            self.assertIn('save_ms', (root / 'module_times.csv').read_text())

    def options(self, root):
        source = root / 'source'
        source.mkdir()
        ok, encoded = cv2.imencode('.jpg', np.zeros((16, 24, 3), dtype=np.uint8))
        self.assertTrue(ok)
        (source / 'frame_000000.jpg').write_bytes(encoded.tobytes())
        return SimpleNamespace(server='http://127.0.0.1:18765', timeout=1, capture_only=False,
            frames_dir=source, interface=None, count=1, fps=1, output_dir=root / 'capture')

    def test_upload_saves_before_post_and_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.options(Path(directory))
            routes = []
            def request(server, route, payload=None, timeout=15):
                routes.append(route)
                if route == '/status':
                    return b'{"received":0,"finished":false}'
                if route == '/frame':
                    self.assertEqual((args.output_dir / 'frame_000000.jpg').read_bytes(), payload)
                    return b'server_frame.jpg'
                return b'finishing'
            with patch.object(send_go2_frames, 'request', request):
                send_go2_frames.run(args)
            self.assertEqual(routes, ['/status', '/frame', '/finish'])
            events = [json.loads(l) for l in (args.output_dir / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['event'], 'complete')

    def test_ambiguous_upload_failure_does_not_retry_or_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.options(Path(directory))
            with patch.object(send_go2_frames, 'request', side_effect=[b'{"received":0}', TimeoutError('upload')]) as request:
                with self.assertRaises(TimeoutError):
                    send_go2_frames.run(args)
            self.assertEqual(request.call_count, 2)
            self.assertTrue((args.output_dir / 'frame_000000.jpg').exists())

    def test_capture_only_needs_no_server_or_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.options(Path(directory))
            args.capture_only = True
            with patch.object(send_go2_frames, 'request') as request:
                send_go2_frames.run(args)
            request.assert_not_called()

    def test_nonempty_session_rejected_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.options(Path(directory))
            with patch.object(send_go2_frames, 'request', return_value=b'{"received":1}'):
                with self.assertRaisesRegex(RuntimeError, 'not fresh'):
                    send_go2_frames.run(args)
            self.assertFalse(args.output_dir.exists())

    def test_receiver_script_parameters_parse_and_bind_loopback(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'live_gaussians_from_rgbd.py').read_text())
        parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
        namespace = dict(argparse=argparse, Path=Path)
        exec(compile(ast.Module(body=[parser], type_ignores=[]), '<parser>', 'exec'), namespace)
        script = (root / 'tools/run_go2_receiver.sh').read_text()
        command = script.split('live_gaussians_from_rgbd.py', 1)[1].split('2>&1', 1)[0]
        command = command.replace('"${GO2_VISUAL_ARGS[@]}"', '')
        with patch.object(sys, 'argv', ['live_gaussians_from_rgbd.py'] + shlex.split(command.replace('\\\n', ' '))):
            args = namespace['parse_args']()
        self.assertEqual(args.stream_host, '127.0.0.1')
        self.assertTrue(args.low_latency_pipeline)
        self.assertTrue(args.preview_noncreation_frames)
        self.assertEqual(args.online_global_optimization_steps, 10)
        self.assertEqual(args.global_optimization_steps, 0)


if __name__ == '__main__':
    unittest.main()
