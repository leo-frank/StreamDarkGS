from __future__ import annotations

import os
import os.path as osp
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .io import export_gaussian_map_state_to_go2dark_ply
from .pi3_preprocess import Pi3RealtimeRunner
from .rgbd import GaussianMapState


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG")


def _natural_sort_key(text: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
        if part
    )


def list_images(image_dir: str) -> list[str]:
    return sorted(
        [
            path.name
            for path in Path(image_dir).iterdir()
            if path.is_file() and path.suffix in IMAGE_EXTENSIONS
        ],
        key=_natural_sort_key,
    )


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().float().clamp(0.0, 1.0)
    if value.dim() == 2:
        value = value.unsqueeze(0)
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    rgb = value[:3].permute(1, 2, 0).numpy()
    return np.ascontiguousarray(rgb[..., ::-1] * 255.0).astype(np.uint8)


def load_state(path: str | Path) -> tuple[GaussianMapState, set[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_payload = payload.get("gaussian_state", payload)
    processed = set(payload.get("processed_names", ()))
    return GaussianMapState.from_dict(state_payload), processed


def load_rgb(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        return transforms.ToTensor()(image.convert("RGB"))


def load_depth(path: str) -> torch.Tensor:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to load depth map: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)


def load_confidence(path: str) -> torch.Tensor:
    confidence = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if confidence is None:
        raise RuntimeError(f"Failed to load confidence map: {path}")
    if confidence.ndim == 3:
        confidence = confidence[..., 0]
    return torch.from_numpy(confidence.astype(np.float32)).unsqueeze(0)


def align_rgbd_to_camera(
    image: torch.Tensor,
    depth: torch.Tensor,
    camera,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = (int(camera.height), int(camera.width))
    if image.shape[-2:] != size:
        image = F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if depth.shape[-2:] != size:
        depth = F.interpolate(depth.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image, depth


def load_aligned_frame_inputs_for_fusion(
    *,
    filename: str,
    image_dir: str,
    source_image_dir: str,
    depth_path: str,
    confidence_path: str | None,
    camera,
    fusion_device: torch.device,
    pi3_runner: Pi3RealtimeRunner,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    cached_images = pi3_runner.get_cached_image_tensors(
        source_image_dir=source_image_dir,
        image_names=[filename],
    )
    image = (
        cached_images[0][1]
        if cached_images
        else load_rgb(osp.join(image_dir, filename))
    )
    depth = load_depth(depth_path)
    confidence = load_confidence(confidence_path) if confidence_path else None
    image, depth = align_rgbd_to_camera(image, depth, camera)
    if confidence is not None and confidence.shape[-2:] != image.shape[-2:]:
        confidence = F.interpolate(
            confidence.unsqueeze(0),
            size=image.shape[-2:],
            mode="nearest",
        ).squeeze(0)
    return (
        image.to(fusion_device),
        depth.to(fusion_device),
        None if confidence is None else confidence.to(fusion_device),
    )


def save_state(
    state: GaussianMapState,
    output_path: Path,
    processed_names: set[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "gaussian_state": {
            key: value.detach().cpu() for key, value in state.as_dict().items()
        },
        "processed_names": sorted(processed_names, key=_natural_sort_key),
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    export_gaussian_map_state_to_go2dark_ply(state, output_path.with_suffix(".ply"))
