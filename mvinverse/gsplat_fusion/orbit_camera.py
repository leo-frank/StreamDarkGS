"""Orbit controls relative to a captured OpenCV camera (+Z forward, +Y down)."""
import math

import torch

from .types import PinholeCamera


def orbit_camera(base: PinholeCamera, distance: float, settings: dict) -> PinholeCamera:
    yaw = float(settings.get('yaw', 0))
    pitch = max(-1.4, min(1.4, float(settings.get('pitch', 0))))
    zoom = max(-2, min(2, float(settings.get('zoom', 0))))
    matrix = base.camera_to_world.detach().cpu().clone()
    rotation = matrix[:3, :3].clone()
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    target = matrix[:3, 3] + rotation[:, 2] * distance
    target += distance * (rotation[:, 0] * settings.get('pan_x', 0) + rotation[:, 1] * settings.get('pan_y', 0))
    matrix[:3, :3] = rotation @ ry @ rx
    matrix[:3, 3] = target - matrix[:3, 2] * distance * math.exp(zoom)
    camera = PinholeCamera(base.image_name, base.width, base.height, base.fx, base.fy, base.cx, base.cy, matrix)
    if settings.get('dragging', 0):
        camera = camera.scaled_to(max(1, base.width // 2), max(1, base.height // 2))
    return camera
