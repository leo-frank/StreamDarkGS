from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ViewerTask:
    version: int
    submitted_at: float
    payload: object


class LatestOnlyViewerWorker:
    """Render previews off the reconstruction path and discard stale queued work."""

    def __init__(self, render: Callable[[ViewerTask], None]) -> None:
        self._render = render
        self._queue: queue.Queue[ViewerTask | None] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._version = 0
        self._closed = False

    def start(self) -> None:
        self._thread.start()

    def submit(self, payload: object) -> tuple[int, bool]:
        self._version += 1
        task = ViewerTask(self._version, time.perf_counter(), payload)
        dropped = False
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            try:
                self._queue.get_nowait()
                dropped = True
            except queue.Empty:
                pass
            self._queue.put_nowait(task)
        return task.version, dropped

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Put the sentinel behind the latest queued task so the final preview is
        # published before the HTTP server is shut down.
        self._queue.put(None)
        self._thread.join(timeout=30.0)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
            started_at = time.perf_counter()
            queue_ms = (started_at - task.submitted_at) * 1000.0
            try:
                self._render(task)
            except Exception as exc:
                print(
                    f"[viewer-timing] version={task.version} failed={exc!r} "
                    f"queue_ms={queue_ms:.1f}",
                    flush=True,
                )
                continue
            render_ms = (time.perf_counter() - started_at) * 1000.0
            print(
                f"[viewer-timing] version={task.version} queue_ms={queue_ms:.1f} "
                f"render_ms={render_ms:.1f}",
                flush=True,
            )
