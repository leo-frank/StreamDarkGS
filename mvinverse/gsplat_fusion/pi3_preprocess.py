from __future__ import annotations

import importlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .types import PinholeCamera
from .pipeline_profiling import StageClock


@dataclass
class Pi3PreprocessOutputs:
    image_dir: Path
    depth_dir: Path
    confidence_dir: Path
    camera_path: Path
    num_frames: int
    image_names: list[str]
    model_input_size: tuple[int, int] | None = None


@dataclass
class GeometryObservation:
    camera: PinholeCamera
    depth: torch.Tensor
    confidence: torch.Tensor
    points_world: np.ndarray


def _natural_sort_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def _orthonormalize_pose(pose: np.ndarray) -> np.ndarray:
    pose = pose.astype(np.float32, copy=True)
    rotation = pose[:3, :3]
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    pose[:3, :3] = rotation
    pose[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return pose


def _estimate_window_scale(
    global_depth_map: dict[str, np.ndarray],
    global_valid_map: dict[str, np.ndarray],
    image_names: list[str],
    local_points: torch.Tensor,
    conf: torch.Tensor,
) -> tuple[float, int]:
    conf_map = conf[..., 0] if conf.dim() == 4 else conf
    candidates: list[float] = []
    for idx, image_name in enumerate(image_names):
        ref_depth = global_depth_map.get(image_name)
        ref_valid = global_valid_map.get(image_name)
        if ref_depth is None or ref_valid is None:
            continue
        cur_depth = local_points[idx, ..., 2].numpy()
        cur_valid = conf_map[idx].numpy() > 0.1
        valid = (
            ref_valid
            & cur_valid
            & np.isfinite(ref_depth)
            & np.isfinite(cur_depth)
            & (ref_depth > 1e-6)
            & (cur_depth > 1e-6)
        )
        if int(valid.sum()) < 64:
            continue
        ratios = ref_depth[valid] / cur_depth[valid]
        ratios = ratios[np.isfinite(ratios) & (ratios > 1e-4)]
        if ratios.size == 0:
            continue
        candidates.append(float(np.median(ratios)))
    if not candidates:
        return 1.0, 0
    scale = float(np.median(np.asarray(candidates, dtype=np.float32)))
    scale = float(np.clip(scale, 0.1, 10.0))
    return scale, len(candidates)


def _estimate_window_alignment(
    global_pose_map: dict[str, np.ndarray],
    image_names: list[str],
    local_camera_poses: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, int]:
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for idx, image_name in enumerate(image_names):
        global_pose = global_pose_map.get(image_name)
        if global_pose is None:
            continue
        local_pose = np.asarray(local_camera_poses[idx], dtype=np.float32)
        rotation = global_pose[:3, :3] @ local_pose[:3, :3].T
        translation = global_pose[:3, 3] - scale * (rotation @ local_pose[:3, 3])
        rotations.append(rotation.astype(np.float32))
        translations.append(translation.astype(np.float32))
    if not rotations:
        return np.eye(4, dtype=np.float32), 0

    rotation_mean = np.stack(rotations, axis=0).mean(axis=0)
    u, _, vt = np.linalg.svd(rotation_mean)
    rotation_aligned = u @ vt
    if np.linalg.det(rotation_aligned) < 0:
        u[:, -1] *= -1.0
        rotation_aligned = u @ vt
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_aligned.astype(np.float32)
    transform[:3, 3] = np.stack(translations, axis=0).mean(axis=0).astype(np.float32)
    return transform, len(rotations)


def _umeyama_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping source points to target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have shape [N, 3]")
    if source.shape[0] < 3:
        raise ValueError("At least three point pairs are required")

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if src_var <= 1e-12:
        raise ValueError("Degenerate source point cloud")

    covariance = (tgt_centered.T @ src_centered) / float(source.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    det = np.linalg.det(u @ vt)
    correction = np.eye(3, dtype=np.float64)
    if det < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / src_var)
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Invalid similarity scale")
    translation = tgt_mean - scale * (rotation @ src_mean)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _camera_points_to_world(
    points_cam: np.ndarray, camera_pose: np.ndarray
) -> np.ndarray:
    pose = np.asarray(camera_pose, dtype=np.float32)
    points = np.asarray(points_cam, dtype=np.float32)
    return points @ pose[:3, :3].T + pose[:3, 3]


def _finite_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values)[np.isfinite(values) & (np.asarray(values) > 0.0)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "p02": None, "p50": None, "p98": None}
    percentiles = np.percentile(finite, [2, 50, 98])
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "p02": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p98": float(percentiles[2]),
    }


def _adjust_points_to_fixed_intrinsics(
    local_points: torch.Tensor,
    predicted_intrinsics: np.ndarray,
    fixed_intrinsics: np.ndarray,
) -> torch.Tensor:
    """Map Pi3 XYZ rays from each predicted K to one fixed K.

    Unlike depth reprojection, this preserves Pi3's original local XYZ structure.
    For an ideal pinhole point it is algebraically equivalent to changing K:
      x' = fx_pred / fx_fixed * x + (cx_pred - cx_fixed) / fx_fixed * z
      y' = fy_pred / fy_fixed * y + (cy_pred - cy_fixed) / fy_fixed * z
    """
    if local_points.dim() != 4 or local_points.shape[-1] != 3:
        raise ValueError("local_points must have shape [N, H, W, 3]")
    predicted = np.asarray(predicted_intrinsics, dtype=np.float32)
    fixed = np.asarray(fixed_intrinsics, dtype=np.float32)
    if predicted.shape != (local_points.shape[0], 3, 3):
        raise ValueError("predicted_intrinsics must have shape [N, 3, 3]")
    fixed_fx, fixed_fy = float(fixed[0, 0]), float(fixed[1, 1])
    fixed_cx, fixed_cy = float(fixed[0, 2]), float(fixed[1, 2])
    if fixed_fx <= 0.0 or fixed_fy <= 0.0:
        raise ValueError("Fixed focal lengths must be positive")
    parameters = torch.as_tensor(
        predicted, dtype=local_points.dtype, device=local_points.device
    )
    z = local_points[..., 2]
    x = (
        parameters[:, 0, 0, None, None] / fixed_fx * local_points[..., 0]
        + (parameters[:, 0, 2, None, None] - fixed_cx) / fixed_fx * z
    )
    y = (
        parameters[:, 1, 1, None, None] / fixed_fy * local_points[..., 1]
        + (parameters[:, 1, 2, None, None] - fixed_cy) / fixed_fy * z
    )
    return torch.stack((x, y, z), dim=-1)


def _estimate_window_pointcloud_alignment(
    global_points_map: dict[str, np.ndarray],
    global_valid_map: dict[str, np.ndarray],
    global_conf_map: dict[str, np.ndarray],
    image_names: list[str],
    local_points: torch.Tensor,
    conf: torch.Tensor,
    local_camera_poses: np.ndarray,
    *,
    max_points: int = 60000,
    min_points: int = 2048,
) -> tuple[np.ndarray, float, float, int, int]:
    """Estimate local-window to global Sim(3) from overlapping same-pixel point clouds."""
    conf_map = conf[..., 0] if conf.dim() == 4 else conf
    source_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    overlap_frames = 0
    per_frame_budget = max(max_points // max(len(image_names), 1), 512)

    for idx, image_name in enumerate(image_names):
        target_world = global_points_map.get(image_name)
        ref_valid = global_valid_map.get(image_name)
        ref_conf = global_conf_map.get(image_name)
        if target_world is None or ref_valid is None:
            continue
        cur_points = local_points[idx].numpy().astype(np.float32, copy=False)
        cur_conf = conf_map[idx].numpy()
        if target_world.shape[:2] != cur_points.shape[:2]:
            continue
        ref_conf_gate = 0.1
        if ref_conf is not None and np.isfinite(ref_conf).any():
            ref_conf_gate = max(ref_conf_gate, float(np.nanmean(ref_conf)) * 0.75)
        cur_conf_gate = (
            max(0.1, float(np.nanmean(cur_conf)) * 0.75)
            if np.isfinite(cur_conf).any()
            else 0.1
        )
        valid = (
            ref_valid
            & np.isfinite(target_world).all(axis=-1)
            & np.isfinite(cur_points).all(axis=-1)
            & np.isfinite(cur_conf)
            & (cur_conf > cur_conf_gate)
            & (cur_points[..., 2] > 1e-6)
        )
        if ref_conf is not None:
            valid = valid & np.isfinite(ref_conf) & (ref_conf > ref_conf_gate)
        valid_indices = np.flatnonzero(valid.reshape(-1))
        if valid_indices.size < 256:
            continue
        if valid_indices.size > per_frame_budget:
            sample_indices = np.linspace(
                0,
                valid_indices.size - 1,
                num=per_frame_budget,
                dtype=np.int64,
            )
            valid_indices = valid_indices[sample_indices]

        cur_flat = cur_points.reshape(-1, 3)[valid_indices]
        tgt_flat = target_world.reshape(-1, 3)[valid_indices]
        src_world = _camera_points_to_world(cur_flat, local_camera_poses[idx])
        source_chunks.append(src_world)
        target_chunks.append(tgt_flat)
        overlap_frames += 1

    if not source_chunks:
        return np.eye(4, dtype=np.float32), 1.0, 1.0, 0, 0

    source = np.concatenate(source_chunks, axis=0)
    target = np.concatenate(target_chunks, axis=0)
    if source.shape[0] > max_points:
        sample_indices = np.linspace(
            0, source.shape[0] - 1, num=max_points, dtype=np.int64
        )
        source = source[sample_indices]
        target = target[sample_indices]
    if source.shape[0] < min_points:
        return (
            np.eye(4, dtype=np.float32),
            1.0,
            1.0,
            overlap_frames,
            int(source.shape[0]),
        )

    try:
        scale, rotation, translation = _umeyama_similarity(source, target)
        residual = np.linalg.norm(
            scale * (source @ rotation.T) + translation[None, :] - target, axis=1
        )
        cutoff = (
            np.quantile(residual[np.isfinite(residual)], 0.8)
            if np.isfinite(residual).any()
            else np.inf
        )
        inlier = np.isfinite(residual) & (residual <= max(float(cutoff), 1e-6))
        fit_source = source
        fit_target = target
        if int(inlier.sum()) >= min_points:
            fit_source = source[inlier]
            fit_target = target[inlier]
            scale, rotation, translation = _umeyama_similarity(fit_source, fit_target)
    except Exception:
        return np.eye(4, dtype=np.float32), 1.0, 1.0, overlap_frames, 0

    raw_scale = float(scale)
    scale = raw_scale
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = translation.astype(np.float32)
    return transform, scale, raw_scale, overlap_frames, int(source.shape[0])


def _window_overlap_residual_stats(
    global_points_map: dict[str, np.ndarray],
    global_valid_map: dict[str, np.ndarray],
    global_conf_map: dict[str, np.ndarray],
    image_names: list[str],
    local_points: torch.Tensor,
    conf: torch.Tensor,
    local_camera_poses: np.ndarray,
    align_to_global: np.ndarray,
    scale: float,
    *,
    max_points: int = 60000,
) -> dict[str, object]:
    """Measure same-pixel overlap errors before and after the fitted Sim(3)."""
    conf_map = conf[..., 0] if conf.dim() == 4 else conf
    rotation = align_to_global[:3, :3].astype(np.float32)
    translation = align_to_global[:3, 3].astype(np.float32)
    per_frame_budget = max(max_points // max(len(image_names), 1), 512)
    before_chunks: list[np.ndarray] = []
    after_chunks: list[np.ndarray] = []
    per_frame: dict[str, dict[str, object]] = {}

    def summarize(values: np.ndarray) -> dict[str, float | int | None]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"pairs": 0, "mean": None, "p50": None, "p80": None,
                    "p90": None, "p95": None}
        percentiles = np.percentile(finite, [50, 80, 90, 95])
        return {
            "pairs": int(finite.size),
            "mean": float(finite.mean()),
            "p50": float(percentiles[0]),
            "p80": float(percentiles[1]),
            "p90": float(percentiles[2]),
            "p95": float(percentiles[3]),
        }

    for index, image_name in enumerate(image_names):
        target = global_points_map.get(image_name)
        ref_valid = global_valid_map.get(image_name)
        ref_conf = global_conf_map.get(image_name)
        if target is None or ref_valid is None:
            continue
        current = local_points[index].numpy().astype(np.float32, copy=False)
        current_conf = conf_map[index].numpy()
        if target.shape[:2] != current.shape[:2]:
            continue
        ref_gate = 0.1
        if ref_conf is not None and np.isfinite(ref_conf).any():
            ref_gate = max(ref_gate, float(np.nanmean(ref_conf)) * 0.75)
        current_gate = (
            max(0.1, float(np.nanmean(current_conf)) * 0.75)
            if np.isfinite(current_conf).any() else 0.1
        )
        valid = (
            ref_valid
            & np.isfinite(target).all(axis=-1)
            & np.isfinite(current).all(axis=-1)
            & np.isfinite(current_conf)
            & (current_conf > current_gate)
            & (current[..., 2] > 1e-6)
        )
        if ref_conf is not None:
            valid &= np.isfinite(ref_conf) & (ref_conf > ref_gate)
        indices = np.flatnonzero(valid.reshape(-1))
        if indices.size > per_frame_budget:
            indices = indices[np.linspace(
                0, indices.size - 1, num=per_frame_budget, dtype=np.int64
            )]
        if indices.size == 0:
            continue
        source = _camera_points_to_world(
            current.reshape(-1, 3)[indices], local_camera_poses[index]
        )
        target_sample = target.reshape(-1, 3)[indices]
        aligned = float(scale) * (source @ rotation.T) + translation[None, :]
        before = np.linalg.norm(source - target_sample, axis=1)
        after = np.linalg.norm(aligned - target_sample, axis=1)
        before_chunks.append(before)
        after_chunks.append(after)
        per_frame[Path(image_name).stem] = {
            "before_alignment": summarize(before),
            "after_alignment": summarize(after),
            "reference_confidence_gate": float(ref_gate),
            "current_confidence_gate": float(current_gate),
        }

    before_all = np.concatenate(before_chunks) if before_chunks else np.empty(0)
    after_all = np.concatenate(after_chunks) if after_chunks else np.empty(0)
    return {
        "before_alignment": summarize(before_all),
        "after_alignment": summarize(after_all),
        "per_frame": per_frame,
    }


def _apply_similarity_to_camera_poses(
    local_camera_poses: np.ndarray,
    align_to_global: np.ndarray,
    scale: float,
) -> np.ndarray:
    rotation_align = align_to_global[:3, :3].astype(np.float32)
    translation_align = align_to_global[:3, 3].astype(np.float32)
    aligned: list[np.ndarray] = []
    for camera_pose in local_camera_poses:
        pose = np.asarray(camera_pose, dtype=np.float32)
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = rotation_align @ pose[:3, :3]
        out[:3, 3] = scale * (rotation_align @ pose[:3, 3]) + translation_align
        aligned.append(_orthonormalize_pose(out))
    return np.stack(aligned, axis=0)


def _images_to_uniform_tensor(
    sources: list[Image.Image],
    image_names: list[str],
    pixel_limit: int = 255000,
) -> tuple[torch.Tensor, float, float, list[str], tuple[int, int]]:
    if not sources:
        raise RuntimeError("No images loaded for Pi3 window")

    first_img = sources[0]
    w_orig, h_orig = first_img.size
    scale = math.sqrt(pixel_limit / (w_orig * h_orig)) if w_orig * h_orig > 0 else 1.0
    w_target, h_target = w_orig * scale, h_orig * scale
    k, m = round(w_target / 14), round(h_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1
    target_w, target_h = max(1, k) * 14, max(1, m) * 14

    to_tensor_transform = transforms.ToTensor()
    tensor_list: list[torch.Tensor] = []
    for img_pil in sources:
        resized_img = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        tensor_list.append(to_tensor_transform(resized_img))
        resized_img.close()

    scale_x = w_orig / target_w
    scale_y = h_orig / target_h
    return (
        torch.stack(tensor_list, dim=0),
        scale_x,
        scale_y,
        image_names,
        (target_h, target_w),
    )


def _run_pi3_model(
    model,
    imgs: torch.Tensor,
    runtime_device: torch.device,
):
    imgs = imgs.to(runtime_device)
    use_amp = runtime_device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.get_device_capability(runtime_device)[0] >= 8
        else torch.float16
    )
    with torch.no_grad():
        with torch.amp.autocast(
            device_type=runtime_device.type, dtype=amp_dtype, enabled=use_amp
        ):
            predictions = model(imgs[None])
    predictions["images"] = imgs[None].permute(0, 1, 3, 4, 2)
    predictions["conf"] = torch.sigmoid(predictions["conf"])
    return predictions


class Pi3GeometryStream:
    def __init__(
        self,
        pi3_root: str | Path,
        ckpt: str = "",
        device: str = "cuda",
        alignment_mode: str = "pointcloud",
        alignment_reference: str = "first",
        overlap_policy: str = "latest",
        capture_window_debug: bool = False,
        fixed_intrinsics_mode: str = "none",
    ) -> None:
        self.pi3_root = Path(pi3_root)
        self.ckpt = ckpt
        self.runtime_device = torch.device(device)
        if alignment_mode not in {"pointcloud", "pose_depth", "none"}:
            raise ValueError(
                "alignment_mode must be one of: pointcloud, pose_depth, none"
            )
        self.alignment_mode = alignment_mode
        if alignment_reference not in {"first", "latest"}:
            raise ValueError("alignment_reference must be first or latest")
        if overlap_policy not in {"first", "latest"}:
            raise ValueError("overlap_policy must be first or latest")
        self.alignment_reference = alignment_reference
        self.overlap_policy = overlap_policy
        self.capture_window_debug = bool(capture_window_debug)
        self.last_window_debug: dict[str, object] | None = None
        if fixed_intrinsics_mode not in {"none", "first_window"}:
            raise ValueError("fixed_intrinsics_mode must be none or first_window")
        self.fixed_intrinsics_mode = fixed_intrinsics_mode
        self.fixed_intrinsics: np.ndarray | None = None
        self.predicted_intrinsics_observations: dict[str, list[np.ndarray]] = {}
        self.geometry_observations: dict[str, list[GeometryObservation]] = {}
        self.Pi3, self.depth_edge, self.recover_focal_shift = _load_pi3_symbols(
            self.pi3_root
        )
        self._image_cache: dict[str, Image.Image] = {}
        self._image_tensor_cache: dict[str, torch.Tensor] = {}
        self._cached_source_dir: Path | None = None
        self._to_tensor = transforms.ToTensor()
        if ckpt:
            self.model = self.Pi3().to(self.runtime_device).eval()
            if ckpt.endswith(".safetensors"):
                load_file = importlib.import_module("safetensors.torch").load_file
                weights = load_file(ckpt)
            else:
                weights = torch.load(
                    ckpt, map_location=self.runtime_device, weights_only=False
                )
            if isinstance(weights, tuple):
                weights = weights[0]
            if isinstance(weights, tuple):
                weights = weights[0]
            if isinstance(weights, dict) and "state_dict" in weights:
                weights = weights["state_dict"]
            self.model.load_state_dict(weights)
        else:
            self.model = (
                self.Pi3.from_pretrained("yyfz233/Pi3").to(self.runtime_device).eval()
            )

    def _get_cached_images(
        self,
        source_image_dir: str | Path,
        image_names: list[str],
    ) -> tuple[list[Image.Image], list[str]]:
        source_image_dir = Path(source_image_dir)
        if self._cached_source_dir != source_image_dir:
            for image in self._image_cache.values():
                image.close()
            self._image_cache.clear()
            self._image_tensor_cache.clear()
            self._cached_source_dir = source_image_dir

        requested = set(image_names)
        stale = [name for name in self._image_cache.keys() if name not in requested]
        for name in stale:
            self._image_cache.pop(name).close()
            self._image_tensor_cache.pop(name, None)

        sources: list[Image.Image] = []
        loaded_names: list[str] = []
        for image_name in image_names:
            cached = self._image_cache.get(image_name)
            if cached is None:
                image_path = source_image_dir / image_name
                try:
                    cached = Image.open(image_path).convert("RGB")
                except Exception as exc:
                    print(f"Could not load image {image_name}: {exc}")
                    continue
                self._image_cache[image_name] = cached
                self._image_tensor_cache[image_name] = self._to_tensor(cached)
            sources.append(cached.copy())
            loaded_names.append(image_name)
        return sources, loaded_names

    def get_cached_image_tensors(
        self,
        source_image_dir: str | Path,
        image_names: list[str],
    ) -> list[tuple[str, torch.Tensor]]:
        self._get_cached_images(source_image_dir, image_names)
        outputs: list[tuple[str, torch.Tensor]] = []
        for image_name in image_names:
            tensor = self._image_tensor_cache.get(image_name)
            if tensor is None:
                continue
            outputs.append((Path(image_name).stem, tensor))
        return outputs

    def _alignment_maps(
        self,
        image_names: list[str],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        poses: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        points: dict[str, np.ndarray] = {}
        confidences: dict[str, np.ndarray] = {}
        valid: dict[str, np.ndarray] = {}
        observation_index = 0 if self.alignment_reference == "first" else -1
        for image_name in image_names:
            observations = self.geometry_observations.get(Path(image_name).stem)
            if not observations:
                continue
            observation = observations[observation_index]
            depth = observation.depth.squeeze(0).numpy()
            confidence = observation.confidence.squeeze(0).numpy()
            poses[image_name] = observation.camera.camera_to_world.numpy()
            depths[image_name] = depth
            points[image_name] = observation.points_world
            confidences[image_name] = confidence
            valid[image_name] = (confidence > 0.1) & np.isfinite(depth) & (depth > 0.0)
        return poses, depths, points, confidences, valid

    def process_window(
        self,
        source_image_dir: str | Path,
        image_names: list[str],
    ) -> tuple[list[str], tuple[int, int]]:
        clock = StageClock(self.runtime_device)
        sources, loaded_names = self._get_cached_images(source_image_dir, image_names)
        try:
            imgs, _, _, loaded_names, input_size = _images_to_uniform_tensor(sources, loaded_names)
        finally:
            for image in sources:
                image.close()
        clock.mark("load_and_preprocess")

        predictions = _run_pi3_model(self.model, imgs, self.runtime_device)
        clock.mark("model")
        edge_mask = self.depth_edge(predictions["local_points"][..., 2], rtol=0.03)
        predictions["conf"][edge_mask] = 0.0
        local_points = predictions["local_points"].squeeze(0).detach().cpu()
        confidence = predictions["conf"].squeeze(0).detach().cpu()
        local_poses = (
            predictions["camera_poses"].squeeze(0).detach().cpu().numpy()
        ).astype(np.float32)
        predicted_intrinsics = _estimate_intrinsics(
            local_points=local_points,
            conf=confidence,
            recover_focal_shift=self.recover_focal_shift,
        )
        if self.fixed_intrinsics_mode == "first_window":
            if self.fixed_intrinsics is None:
                self.fixed_intrinsics = np.median(
                    predicted_intrinsics, axis=0
                ).astype(np.float32)
                self.fixed_intrinsics[2, :] = np.array(
                    [0.0, 0.0, 1.0], dtype=np.float32
                )
                print(
                    "[pi3] fixed intrinsics initialized from first window: "
                    f"fx={self.fixed_intrinsics[0, 0]:.4f} "
                    f"fy={self.fixed_intrinsics[1, 1]:.4f} "
                    f"cx={self.fixed_intrinsics[0, 2]:.4f} "
                    f"cy={self.fixed_intrinsics[1, 2]:.4f}",
                    flush=True,
                )
            local_points = _adjust_points_to_fixed_intrinsics(
                local_points, predicted_intrinsics, self.fixed_intrinsics
            )
        clock.mark("edge_filter_and_cpu_transfer")

        poses, depths, points, confidences, valid = self._alignment_maps(loaded_names)
        transform = np.eye(4, dtype=np.float32)
        scale = 1.0
        raw_scale = 1.0
        overlap_count = 0
        point_pairs = 0
        scale_overlap = 0
        mode = self.alignment_mode
        if self.alignment_mode == "pointcloud":
            transform, scale, raw_scale, overlap_count, point_pairs = (
                _estimate_window_pointcloud_alignment(
                    points,
                    valid,
                    confidences,
                    loaded_names,
                    local_points,
                    confidence,
                    local_poses,
                )
            )
            scale_overlap = overlap_count
            if overlap_count <= 0 or point_pairs < 2048:
                scale, scale_overlap = _estimate_window_scale(
                    depths, valid, loaded_names, local_points, confidence
                )
                raw_scale = scale
                transform, overlap_count = _estimate_window_alignment(
                    poses, loaded_names, local_poses, scale=scale
                )
                mode = "pointcloud->pose-depth" if overlap_count else "init-window"
                point_pairs = 0
        elif self.alignment_mode == "pose_depth":
            scale, scale_overlap = _estimate_window_scale(
                depths, valid, loaded_names, local_points, confidence
            )
            raw_scale = scale
            transform, overlap_count = _estimate_window_alignment(
                poses, loaded_names, local_poses, scale=scale
            )
            mode = "pose-depth" if overlap_count else "init-window"

        clock.mark("window_alignment")
        points_before_alignment: list[np.ndarray] | None = None
        overlap_residuals: dict[str, object] | None = None
        depth_before_alignment: list[dict[str, float | int | None]] | None = None
        if self.capture_window_debug:
            points_before_alignment = [
                _camera_points_to_world(
                    local_points[index].numpy().astype(np.float32, copy=False),
                    local_poses[index],
                ).astype(np.float32)
                for index in range(len(loaded_names))
            ]
            depth_before_alignment = [
                _finite_distribution(local_points[index, ..., 2].numpy())
                for index in range(len(loaded_names))
            ]
            overlap_residuals = _window_overlap_residual_stats(
                points,
                valid,
                confidences,
                loaded_names,
                local_points,
                confidence,
                local_poses,
                transform,
                scale,
            )
        if abs(scale - 1.0) > 1e-6:
            local_points = local_points * float(scale)
        camera_poses = _apply_similarity_to_camera_poses(local_poses, transform, scale)
        if self.fixed_intrinsics is None:
            intrinsics = predicted_intrinsics
        else:
            intrinsics = np.repeat(
                self.fixed_intrinsics[None, ...], len(loaded_names), axis=0
            )
        clock.mark("transform_and_intrinsics")
        height, width = input_size
        confidence_maps = confidence[..., 0] if confidence.dim() == 4 else confidence
        points_after_alignment: list[np.ndarray] = []
        frame_debug: list[dict[str, object]] = []
        for index, image_name in enumerate(loaded_names):
            stem = Path(image_name).stem
            previous_observations = self.geometry_observations.get(stem, [])
            previous_predicted_intrinsics = self.predicted_intrinsics_observations.get(
                stem, []
            )
            reference_observation = None
            if previous_observations:
                reference_index = 0 if self.alignment_reference == "first" else -1
                reference_observation = previous_observations[reference_index]
            camera = PinholeCamera(
                image_name=stem,
                width=width,
                height=height,
                fx=float(intrinsics[index, 0, 0]),
                fy=float(intrinsics[index, 1, 1]),
                cx=float(intrinsics[index, 0, 2]),
                cy=float(intrinsics[index, 1, 2]),
                camera_to_world=torch.from_numpy(camera_poses[index].copy()),
            )
            points_world = _camera_points_to_world(
                local_points[index].numpy().astype(np.float32, copy=False),
                camera_poses[index],
            ).astype(np.float32)
            if self.capture_window_debug:
                points_after_alignment.append(points_world.copy())
                intrinsic_delta = None
                if reference_observation is not None:
                    reference_camera = reference_observation.camera
                    intrinsic_delta = {
                        "reference_fx": float(reference_camera.fx),
                        "reference_fy": float(reference_camera.fy),
                        "reference_cx": float(reference_camera.cx),
                        "reference_cy": float(reference_camera.cy),
                        "fx_delta": float(camera.fx - reference_camera.fx),
                        "fy_delta": float(camera.fy - reference_camera.fy),
                        "cx_delta": float(camera.cx - reference_camera.cx),
                        "cy_delta": float(camera.cy - reference_camera.cy),
                        "fx_relative": float(camera.fx / reference_camera.fx - 1.0),
                        "fy_relative": float(camera.fy / reference_camera.fy - 1.0),
                    }
                frame_debug.append(
                    {
                        "image_name": stem,
                        "intrinsics": {
                            "fx": float(camera.fx), "fy": float(camera.fy),
                            "cx": float(camera.cx), "cy": float(camera.cy),
                            "width": int(camera.width), "height": int(camera.height),
                        },
                        "intrinsics_vs_reference": intrinsic_delta,
                        "pi3_predicted_intrinsics": {
                            "fx": float(predicted_intrinsics[index, 0, 0]),
                            "fy": float(predicted_intrinsics[index, 1, 1]),
                            "cx": float(predicted_intrinsics[index, 0, 2]),
                            "cy": float(predicted_intrinsics[index, 1, 2]),
                        },
                        "pi3_predicted_intrinsics_vs_reference": (
                            None if not previous_predicted_intrinsics else {
                                "reference_fx": float(previous_predicted_intrinsics[reference_index][0, 0]),
                                "reference_fy": float(previous_predicted_intrinsics[reference_index][1, 1]),
                                "fx_delta": float(
                                    predicted_intrinsics[index, 0, 0]
                                    - previous_predicted_intrinsics[reference_index][0, 0]
                                ),
                                "fy_delta": float(
                                    predicted_intrinsics[index, 1, 1]
                                    - previous_predicted_intrinsics[reference_index][1, 1]
                                ),
                                "fx_relative": float(
                                    predicted_intrinsics[index, 0, 0]
                                    / previous_predicted_intrinsics[reference_index][0, 0] - 1.0
                                ),
                                "fy_relative": float(
                                    predicted_intrinsics[index, 1, 1]
                                    / previous_predicted_intrinsics[reference_index][1, 1] - 1.0
                                ),
                            }
                        ),
                        "camera_to_world_before_alignment": local_poses[index].tolist(),
                        "camera_to_world_after_alignment": camera_poses[index].tolist(),
                        "depth_before_alignment": depth_before_alignment[index],
                        "depth_after_alignment": _finite_distribution(
                            local_points[index, ..., 2].numpy()
                        ),
                    }
                )
            observation = GeometryObservation(
                camera=camera,
                depth=local_points[index, ..., 2].float().unsqueeze(0),
                confidence=confidence_maps[index].float().unsqueeze(0),
                points_world=points_world,
            )
            self.geometry_observations.setdefault(stem, []).append(observation)
            self.predicted_intrinsics_observations.setdefault(stem, []).append(
                predicted_intrinsics[index].copy()
            )

        if self.capture_window_debug:
            self.last_window_debug = {
                "image_names": list(loaded_names),
                "points_before_alignment": points_before_alignment,
                "points_after_alignment": points_after_alignment,
                "confidence": confidence_maps.numpy().astype(np.float32, copy=True),
                "transform": transform.copy(),
                "scale": float(scale),
                "raw_scale": float(raw_scale),
                "overlap_count": int(overlap_count),
                "point_pairs": int(point_pairs),
                "mode": mode,
                "frames": frame_debug,
                "overlap_residuals": overlap_residuals,
                "fixed_intrinsics_mode": self.fixed_intrinsics_mode,
                "fixed_intrinsics_adjustment": (
                    "xyz_ray_transform"
                    if self.fixed_intrinsics is not None else "none"
                ),
                "fixed_intrinsics": (
                    None if self.fixed_intrinsics is None
                    else self.fixed_intrinsics.tolist()
                ),
            }

        clock.mark("store_observations")
        clock.report("pi3", first=loaded_names[0], frames=len(loaded_names))

        print(
            f"[pi3] alignment overlap={overlap_count} scale_overlap={scale_overlap} "
            f"point_pairs={point_pairs} raw_scale={raw_scale:.4f} "
            f"scale={scale:.4f} mode={mode}",
            flush=True,
        )
        return loaded_names, input_size

    def resolve_frame(self, stem: str) -> tuple[GeometryObservation, int]:
        observations = self.geometry_observations.get(stem)
        if not observations:
            raise KeyError(f"No Pi3 geometry observations for {stem}")
        index = 0 if self.overlap_policy == "first" else -1
        return observations[index], len(observations)

    def release_frame(self, stem: str) -> None:
        self.geometry_observations.pop(stem, None)
        self.predicted_intrinsics_observations.pop(stem, None)


Pi3RealtimeRunner = Pi3GeometryStream


def _ensure_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5
    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def _solve_optimal_focal_shift(
    uv: np.ndarray, xyz: np.ndarray
) -> tuple[np.float32, np.float32]:
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def residual(shift: np.ndarray) -> np.ndarray:
        xy_proj = xy / (z + shift)[:, None]
        focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        return (focal * xy_proj - uv).ravel()

    solution = least_squares(residual, x0=0, ftol=1e-3, method="lm")
    shift = solution.x.squeeze().astype(np.float32)
    xy_proj = xy / (z + shift)[:, None]
    focal = ((xy_proj * uv).sum() / np.square(xy_proj).sum()).astype(np.float32)
    return shift, focal


def _solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float) -> np.float32:
    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def residual(shift: np.ndarray) -> np.ndarray:
        xy_proj = xy / (z + shift)[:, None]
        return (focal * xy_proj - uv).ravel()

    solution = least_squares(residual, x0=0, ftol=1e-3, method="lm")
    return solution.x.squeeze().astype(np.float32)


def _recover_focal_shift(
    points: torch.Tensor,
    mask: torch.Tensor | None = None,
    focal: torch.Tensor | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover normalized focal length and Z shift from a dense point map."""
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None

    uv = _normalized_view_plane_uv(
        width, height, dtype=points.dtype, device=points.device
    )
    points_lr = F.interpolate(
        points.permute(0, 3, 1, 2),
        downsample_size,
        mode="nearest",
    ).permute(0, 2, 3, 1)
    uv_lr = (
        F.interpolate(
            uv.unsqueeze(0).permute(0, 3, 1, 2),
            downsample_size,
            mode="nearest",
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    mask_lr = (
        None
        if mask is None
        else F.interpolate(
            mask.to(torch.float32).unsqueeze(1),
            downsample_size,
            mode="nearest",
        ).squeeze(1)
        > 0
    )

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift: list[float] = []
    optim_focal: list[float] = []
    for idx in range(points.shape[0]):
        points_i = (
            points_lr_np[idx] if mask is None else points_lr_np[idx][mask_lr_np[idx]]
        )
        uv_i = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[idx]]
        if uv_i.shape[0] < 2:
            optim_focal.append(1.0)
            optim_shift.append(0.0)
            continue
        if focal is None:
            shift_i, focal_i = _solve_optimal_focal_shift(uv_i, points_i)
            optim_focal.append(float(focal_i))
        else:
            shift_i = _solve_optimal_shift(uv_i, points_i, focal_np[idx])
        optim_shift.append(float(shift_i))

    shift_tensor = torch.tensor(
        optim_shift,
        device=points.device,
        dtype=points.dtype,
    ).reshape(shape[:-3])
    if focal is None:
        focal_tensor = torch.tensor(
            optim_focal,
            device=points.device,
            dtype=points.dtype,
        ).reshape(shape[:-3])
    else:
        focal_tensor = focal.reshape(shape[:-3])
    return focal_tensor, shift_tensor


def _load_pi3_symbols(pi3_root: Path):
    _ensure_sys_path(pi3_root)
    _ensure_sys_path(pi3_root / "utils3d")

    Pi3 = importlib.import_module("pi3.models.pi3").Pi3
    depth_edge = importlib.import_module("pi3.utils.geometry").depth_edge
    return Pi3, depth_edge, _recover_focal_shift


def _estimate_intrinsics(
    local_points: torch.Tensor, conf: torch.Tensor, recover_focal_shift
) -> np.ndarray:
    if local_points.dim() != 4 or local_points.shape[-1] != 3:
        raise ValueError(
            f"local_points must have shape [N, H, W, 3], got {tuple(local_points.shape)}"
        )
    masks = conf[..., 0] > 0.1
    focal, _shift = recover_focal_shift(
        local_points[None], mask=masks[None], downsample_size=(64, 64)
    )
    focal = focal.squeeze(0).to(dtype=torch.float32)

    height = int(local_points.shape[-3])
    width = int(local_points.shape[-2])
    aspect_ratio = float(width) / float(height)
    diag_factor = math.sqrt(1.0 + aspect_ratio**2)

    fx = focal / 2.0 * diag_factor / aspect_ratio * float(width)
    fy = focal / 2.0 * diag_factor * float(height)
    cx = torch.full_like(fx, float(width) / 2.0)
    cy = torch.full_like(fy, float(height) / 2.0)

    intrinsics = torch.zeros((local_points.shape[0], 3, 3), dtype=torch.float32)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0
    return intrinsics.cpu().numpy()
