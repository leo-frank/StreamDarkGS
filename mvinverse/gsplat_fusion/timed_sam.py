from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable

import torch


class TimedSAMMaskGenerator:
    """Transparent timing wrapper for SAM2AutomaticMaskGenerator."""

    def __init__(self, generator: object) -> None:
        self.generator = generator
        self.last_timings_ms: dict[str, float] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.generator, name)

    def _device(self) -> torch.device:
        return torch.device(self.generator.predictor.device)

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def generate(self, image) -> list[dict[str, Any]]:
        device = self._device()
        totals: defaultdict[str, float] = defaultdict(float)
        counts: defaultdict[str, int] = defaultdict(int)
        originals: list[tuple[object, str, Callable[..., Any]]] = []

        def install(owner: object, name: str, timing_name: str) -> None:
            original = getattr(owner, name)
            originals.append((owner, name, original))

            def timed(*args, **kwargs):
                self._sync(device)
                started_at = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    self._sync(device)
                    totals[timing_name] += (time.perf_counter() - started_at) * 1000.0
                    counts[timing_name] += 1

            setattr(owner, name, timed)

        predictor = self.generator.predictor
        install(self.generator, "_generate_masks", "generate_masks_ms")
        install(self.generator, "_process_crop", "crop_total_ms")
        install(self.generator, "_process_batch", "batch_total_ms")
        install(predictor, "set_image", "set_image_ms")
        install(predictor, "_predict", "predict_ms")
        install(predictor, "reset_predictor", "reset_predictor_ms")

        self._sync(device)
        started_at = time.perf_counter()
        try:
            result = self.generator.generate(image)
        finally:
            self._sync(device)
            total_ms = (time.perf_counter() - started_at) * 1000.0
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)

        generate_masks_ms = totals["generate_masks_ms"]
        batch_total_ms = totals["batch_total_ms"]
        predict_ms = totals["predict_ms"]
        crop_total_ms = totals["crop_total_ms"]
        set_image_ms = totals["set_image_ms"]
        reset_ms = totals["reset_predictor_ms"]
        self.last_timings_ms = {
            "total_ms": total_ms,
            "generate_masks_ms": generate_masks_ms,
            "encode_records_ms": max(total_ms - generate_masks_ms, 0.0),
            "set_image_ms": set_image_ms,
            "predict_ms": predict_ms,
            "batch_total_ms": batch_total_ms,
            "batch_postprocess_ms": max(batch_total_ms - predict_ms, 0.0),
            "crop_total_ms": crop_total_ms,
            "crop_postprocess_ms": max(
                crop_total_ms - set_image_ms - batch_total_ms - reset_ms, 0.0
            ),
            "reset_predictor_ms": reset_ms,
            "crop_count": float(counts["crop_total_ms"]),
            "batch_count": float(counts["batch_total_ms"]),
            "predict_count": float(counts["predict_ms"]),
        }
        return result
