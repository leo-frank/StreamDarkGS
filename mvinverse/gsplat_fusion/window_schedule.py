from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowStep:
    index: int
    start: int
    frame_names: tuple[str, ...]
    mature_names: tuple[str, ...]
    is_final: bool


def build_window_schedule(
    frame_names: list[str] | tuple[str, ...],
    *,
    window_size: int,
    window_stride: int,
) -> tuple[WindowStep, ...]:
    names = tuple(frame_names)
    size = int(window_size)
    stride = int(window_stride)
    if size < 1:
        raise ValueError("window_size must be positive")
    if stride < 1 or stride > size:
        raise ValueError("window_stride must be in [1, window_size]")
    if not names:
        return ()

    starts: list[int] = []
    start = 0
    while True:
        starts.append(start)
        if start + size >= len(names):
            break
        start += stride

    steps: list[WindowStep] = []
    for index, start in enumerate(starts):
        is_final = index == len(starts) - 1
        mature_end = len(names) if is_final else starts[index + 1]
        steps.append(
            WindowStep(
                index=index,
                start=start,
                frame_names=names[start : start + size],
                mature_names=names[start:mature_end],
                is_final=is_final,
            )
        )
    return tuple(steps)
