from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .types import GaussianMaterialState, PinholeCamera


def _frame_to_camera(frame: dict) -> PinholeCamera:
    image_name = Path(frame.get("image_name", frame.get("img_name", frame.get("file_path", "")))).stem
    if not image_name:
        raise KeyError("Could not resolve image name from camera frame")

    camera_to_world = frame.get("camera_to_world", frame.get("c2w_4x4", frame.get("transform_matrix")))
    if camera_to_world is None:
        raise KeyError(f"Camera frame for {image_name} is missing camera_to_world/transform_matrix")
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    # Match go2_dark Pi3 reader: transforms.json stores NeRF/OpenGL-style c2w,
    # which needs Y/Z axis flips before converting to w2c / world space.
    if "transform_matrix" in frame:
        camera_to_world = camera_to_world.copy()
        camera_to_world[:3, 1:3] *= -1.0

    width = int(frame.get("width", frame.get("w")))
    height = int(frame.get("height", frame.get("h")))
    fx = float(frame.get("fx", frame.get("fl_x")))
    fy = float(frame.get("fy", frame.get("fl_y")))
    cx = float(frame.get("cx"))
    cy = float(frame.get("cy"))

    return PinholeCamera(
        image_name=image_name,
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_to_world=torch.from_numpy(camera_to_world),
    )


def _load_pose_override(path: str | Path | None) -> dict[str, torch.Tensor]:
    if not path:
        return {}
    override_path = Path(path)
    with override_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("cameras", payload if isinstance(payload, list) else [])
    overrides: dict[str, torch.Tensor] = {}
    for row in rows:
        image_name = Path(row.get("img_name", row.get("image_name", ""))).stem
        c2w = row.get("c2w_4x4", row.get("camera_to_world", row.get("transform_matrix")))
        if image_name and c2w is not None:
            overrides[image_name] = torch.tensor(c2w, dtype=torch.float32)
    return overrides


def load_camera_manifest(path: str | Path, pose_override_path: str | Path | None = None) -> dict[str, PinholeCamera]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload["frames"] if isinstance(payload, dict) and "frames" in payload else payload
    overrides = _load_pose_override(pose_override_path)
    cameras: dict[str, PinholeCamera] = {}
    for frame in frames:
        camera = _frame_to_camera(frame)
        if camera.image_name in overrides:
            camera = PinholeCamera(
                image_name=camera.image_name,
                width=camera.width,
                height=camera.height,
                fx=camera.fx,
                fy=camera.fy,
                cx=camera.cx,
                cy=camera.cy,
                camera_to_world=overrides[camera.image_name],
            )
        cameras[camera.image_name] = camera
    if not cameras:
        raise RuntimeError(f"No cameras found in {manifest_path}")
    return cameras


def _load_binary_ply_xyz(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        num_vertices = None
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF while reading PLY header: {path}")
            line_str = line.decode("latin1").strip()
            if line_str.startswith("element vertex"):
                num_vertices = int(line_str.split()[-1])
            if line_str == "end_header":
                break

        if num_vertices is None:
            raise KeyError(f"PLY file is missing vertex count: {path}")

        xyz = np.empty((num_vertices, 3), dtype=np.float32)
        record_struct = struct.Struct("<fffBBBf")
        for idx in range(num_vertices):
            chunk = f.read(record_struct.size)
            if len(chunk) != record_struct.size:
                raise RuntimeError(f"Unexpected EOF while reading PLY vertices from {path}")
            xyz[idx] = record_struct.unpack(chunk)[:3]
    return torch.from_numpy(xyz)


def load_gaussian_means(path: str | Path) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
    elif suffix == ".npz":
        payload = dict(np.load(path))
    elif suffix == ".ply":
        return _load_binary_ply_xyz(path).float()
    else:
        raise ValueError(f"Unsupported gaussian file: {path}")

    if isinstance(payload, dict) and "splats" in payload and "means" in payload["splats"]:
        return torch.as_tensor(payload["splats"]["means"], dtype=torch.float32)

    for key in ("means", "means_world", "xyz", "_xyz"):
        if key in payload:
            return torch.as_tensor(payload[key], dtype=torch.float32)
    raise KeyError(f"Could not find gaussian centers in {path}; tried means/means_world/xyz/_xyz")


def save_gaussian_material_state(state: GaussianMaterialState, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state.as_dict(), output_path)


def _safe_logit(values: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.logit(values.clamp(min=eps, max=1.0 - eps))


def _quats_from_normals(normals_world: torch.Tensor) -> torch.Tensor:
    normals_world = F.normalize(normals_world, dim=-1, eps=1e-6)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=normals_world.device, dtype=normals_world.dtype).expand_as(normals_world)
    dots = (z_axis * normals_world).sum(dim=-1, keepdim=True)
    xyz = torch.cross(z_axis, normals_world, dim=-1)
    quats = torch.cat([1.0 + dots, xyz], dim=-1)

    opposite = dots.squeeze(-1) < -0.9999
    if opposite.any():
        fallback = torch.zeros((int(opposite.sum().item()), 4), device=normals_world.device, dtype=normals_world.dtype)
        fallback[:, 2] = 1.0
        quats[opposite] = fallback

    small = quats.norm(dim=-1) < 1e-8
    if small.any():
        quats[small, 0] = 1.0
        quats[small, 1:] = 0.0
    return F.normalize(quats, dim=-1, eps=1e-6)


def _rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def _write_binary_vertex_ply(path: Path, property_specs: list[tuple[str, str]], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    type_map = {
        "f4": "float",
        "u1": "uchar",
    }
    with path.open("wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {rows.shape[0]}\n".encode("ascii"))
        for name, dtype in property_specs:
            f.write(f"property {type_map[dtype]} {name}\n".encode("ascii"))
        f.write(b"end_header\n")
        rows.tofile(f)


def _write_store_ply_rgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8, copy=False)
    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    rows = np.empty(xyz.shape[0], dtype=dtype)
    rows["x"] = xyz[:, 0]
    rows["y"] = xyz[:, 1]
    rows["z"] = xyz[:, 2]
    rows["nx"] = normals[:, 0]
    rows["ny"] = normals[:, 1]
    rows["nz"] = normals[:, 2]
    rows["red"] = rgb[:, 0]
    rows["green"] = rgb[:, 1]
    rows["blue"] = rgb[:, 2]
    _write_binary_vertex_ply(path, dtype, rows)


def export_gaussian_map_state_to_go2dark_ply(
    state,
    path: str | Path,
    sh_degree: int = 3,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(state.means_world.shape[0])
    sh_coeff_count = (sh_degree + 1) ** 2
    f_rest_dim = 3 * max(sh_coeff_count - 1, 0)

    xyz = state.means_world.detach().cpu().to(dtype=torch.float32)
    normals = torch.zeros_like(xyz)
    diffuse = state.albedo.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0)
    f_dc = _rgb_to_sh_dc(diffuse)
    f_rest = torch.zeros((n, f_rest_dim), dtype=torch.float32)
    opacity = _safe_logit(state.opacities.detach().cpu().to(dtype=torch.float32))
    scale = torch.log(state.scales.detach().cpu().to(dtype=torch.float32).clamp_min(1e-8))
    rotation = _quats_from_normals(state.normals_world.detach().cpu().to(dtype=torch.float32))
    diffuse_raw = _safe_logit(diffuse)
    roughness_raw = _safe_logit(state.roughness.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0))
    metallic_raw = _safe_logit(state.metallic.detach().cpu().to(dtype=torch.float32).clamp(0.0, 1.0))

    property_specs: list[tuple[str, str]] = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ]
    property_specs.extend((f"f_dc_{i}", "f4") for i in range(3))
    property_specs.extend((f"f_rest_{i}", "f4") for i in range(f_rest_dim))
    property_specs.append(("opacity", "f4"))
    property_specs.extend((f"scale_{i}", "f4") for i in range(3))
    property_specs.extend((f"rot_{i}", "f4") for i in range(4))
    property_specs.extend((f"diffuse_{i}", "f4") for i in range(3))
    property_specs.append(("roughness", "f4"))
    property_specs.append(("metallic", "f4"))

    dtype = [(name, dtype_name) for name, dtype_name in property_specs]
    rows = np.empty(n, dtype=dtype)
    rows["x"] = xyz[:, 0].numpy()
    rows["y"] = xyz[:, 1].numpy()
    rows["z"] = xyz[:, 2].numpy()
    rows["nx"] = normals[:, 0].numpy()
    rows["ny"] = normals[:, 1].numpy()
    rows["nz"] = normals[:, 2].numpy()
    for i in range(3):
        rows[f"f_dc_{i}"] = f_dc[:, i].numpy()
    for i in range(f_rest_dim):
        rows[f"f_rest_{i}"] = f_rest[:, i].numpy()
    rows["opacity"] = opacity[:, 0].numpy()
    for i in range(3):
        rows[f"scale_{i}"] = scale[:, i].numpy()
    for i in range(4):
        rows[f"rot_{i}"] = rotation[:, i].numpy()
    for i in range(3):
        rows[f"diffuse_{i}"] = diffuse_raw[:, i].numpy()
    rows["roughness"] = roughness_raw[:, 0].numpy()
    rows["metallic"] = metallic_raw[:, 0].numpy()

    _write_binary_vertex_ply(output_path, property_specs, rows)

    xyz_np = xyz.numpy()
    diffuse_rgb = (diffuse.numpy() * 255.0).astype(np.float32, copy=False)
    fd_rgb = (((1.0 - state.metallic.detach().cpu().numpy()) * diffuse.numpy()) / np.pi * 255.0).astype(np.float32, copy=False)
    _write_store_ply_rgb(output_path.parent / "diffuse.ply", xyz_np, diffuse_rgb)
    _write_store_ply_rgb(output_path.parent / "fd.ply", xyz_np, fd_rgb)
    return output_path
