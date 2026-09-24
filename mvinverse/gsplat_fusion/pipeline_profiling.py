"""Opt-in stage clocks. Sync mode measures GPU-complete wall time, not kernel time."""
from __future__ import annotations

import os
import time

import torch


class StageClock:
    def __init__(self, device=None):
        mode = os.environ.get("STREAMDARKGS_PROFILE_TIMING", "off")
        self.enabled = mode in {"wall", "sync"}
        self.device = torch.device(device) if device is not None else None
        self.synchronize = self.enabled and mode == "sync" and self.device is not None and self.device.type == "cuda"
        self.values = {}
        self._sync()
        self.start = self.previous = time.perf_counter()

    def _sync(self):
        if self.synchronize:
            torch.cuda.synchronize(self.device)

    def mark(self, name):
        if not self.enabled:
            return
        self._sync()
        now = time.perf_counter()
        self.values[name + "_ms"] = (now - self.previous) * 1000.0
        self.previous = now

    def report(self, tag, **metadata):
        if not self.enabled:
            return
        fields = " ".join(f"{key}={value}" for key, value in metadata.items())
        times = " ".join(f"{key}={value:.3f}" for key, value in self.values.items())
        print(f"[profile-{tag}] {fields} {times}", flush=True)


class PreviewSink:
    """Headless publication target; inference, snapshots, rendering and JPEGs run normally.

    No HTTP/network/display time is simulated. Used only for offline benchmarks.
    """
    def viewer_settings(self):
        return {"light_x": -0.25, "light_y": -0.35, "ambient": 0.05, "revision": 0}

    def publish_previews(self, **kwargs):
        pass

    def publish_predictions(self, **kwargs):
        pass

    def close(self):
        pass
