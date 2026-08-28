from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .io import load_camera_manifest
from .rgbd import GaussianMapState, normals_from_quats
from .types import GaussianMaterialState, PinholeCamera


def ensure_local_gsplat_path(gsplat_root: str | Path | None = None) -> Path:
    if gsplat_root is None:
        gsplat_root = (
            Path(__file__).resolve().parents[2] / "third_party" / "gsplat_legacy_cu118"
        )
    gsplat_root = Path(gsplat_root).resolve()
    if not gsplat_root.is_dir():
        raise FileNotFoundError(f"gsplat root not found: {gsplat_root}")
    root_str = str(gsplat_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return gsplat_root


def load_gsplat_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[torch.nn.ParameterDict, dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=device)

    if isinstance(payload, dict) and "splats" in payload:
        splat_state = payload["splats"]
        metadata = {k: v for k, v in payload.items() if k != "splats"}
    elif isinstance(payload, dict) and all(
        k in payload for k in ("means", "scales", "quats", "opacities")
    ):
        splat_state = payload
        metadata = {}
    else:
        raise KeyError(
            f"Unsupported gsplat checkpoint format in {checkpoint_path}. "
            "Expected {'splats': ...} or a bare splat state dict."
        )

    splats = torch.nn.ParameterDict(
        {
            key: torch.nn.Parameter(torch.as_tensor(value, device=device).clone())
            for key, value in splat_state.items()
        }
    )
    return splats, metadata


def attach_material_signal(
    splats: torch.nn.ParameterDict,
    material_state: GaussianMaterialState,
    scene_id: str = "scene",
    gsplat_root: str | Path | None = None,
):
    ensure_local_gsplat_path(gsplat_root)
    from gsplat.scene import GaussianScene

    means = splats["means"].detach()
    if means.shape[0] != material_state.means_world.shape[0]:
        raise ValueError(
            f"Material gaussian count {material_state.means_world.shape[0]} does not match "
            f"gsplat checkpoint {means.shape[0]}"
        )

    signal = {
        "albedo": material_state.albedo.to(device=means.device, dtype=means.dtype),
        "roughness": material_state.roughness.to(
            device=means.device, dtype=means.dtype
        ),
        "metallic": material_state.metallic.to(device=means.device, dtype=means.dtype),
        "normal_world": material_state.normal_world.to(
            device=means.device, dtype=means.dtype
        ),
        "confidence_sum": material_state.confidence_sum.to(
            device=means.device, dtype=means.dtype
        ),
        "update_count": material_state.update_count.to(
            device=means.device, dtype=means.dtype
        ),
    }
    if material_state.material_latent is not None:
        signal["material_latent"] = material_state.material_latent.to(
            device=means.device, dtype=means.dtype
        )
    return GaussianScene.from_splats(splats, id=scene_id, signal=signal)


def _make_viewmat(
    camera: PinholeCamera, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    world_to_camera = torch.linalg.inv(
        camera.camera_to_world.to(device=device, dtype=dtype)
    )
    return world_to_camera


def _make_intrinsics(
    camera: PinholeCamera, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    K = torch.eye(3, device=device, dtype=dtype)
    K[0, 0] = camera.fx
    K[1, 1] = camera.fy
    K[0, 2] = camera.cx
    K[1, 2] = camera.cy
    return K


def _project_world_to_image(
    means_world: torch.Tensor,
    camera: PinholeCamera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_cpu = camera.to(torch.device("cpu"))
    points_world = means_world.detach().to(
        device=torch.device("cpu"), dtype=torch.float32
    )
    world_to_camera = camera_cpu.world_to_camera
    points_camera = (
        points_world @ world_to_camera[:3, :3].transpose(0, 1) + world_to_camera[:3, 3]
    )

    z = points_camera[:, 2]
    valid = torch.isfinite(points_camera).all(dim=-1) & (z > 1e-6)
    if not valid.any():
        empty_int = np.zeros((0,), dtype=np.int32)
        empty_float = np.zeros((0,), dtype=np.float32)
        empty_index = np.zeros((0,), dtype=np.int64)
        return empty_int, empty_int, empty_float, empty_index

    points_camera = points_camera[valid]
    z = points_camera[:, 2]
    u = camera_cpu.fx * (points_camera[:, 0] / z) + camera_cpu.cx
    v = camera_cpu.fy * (points_camera[:, 1] / z) + camera_cpu.cy

    x = torch.round(u).to(torch.int64)
    y = torch.round(v).to(torch.int64)
    in_bounds = (x >= 0) & (x < camera_cpu.width) & (y >= 0) & (y < camera_cpu.height)
    if not in_bounds.any():
        empty_int = np.zeros((0,), dtype=np.int32)
        empty_float = np.zeros((0,), dtype=np.float32)
        empty_index = np.zeros((0,), dtype=np.int64)
        return empty_int, empty_int, empty_float, empty_index

    x = x[in_bounds].cpu().numpy().astype(np.int32, copy=False)
    y = y[in_bounds].cpu().numpy().astype(np.int32, copy=False)
    z = z[in_bounds].cpu().numpy().astype(np.float32, copy=False)
    valid_indices = (
        valid.nonzero(as_tuple=False)
        .squeeze(-1)[in_bounds]
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )
    return x, y, z, valid_indices


def _render_projected_attribute(
    values: torch.Tensor,
    means_world: torch.Tensor,
    camera: PinholeCamera,
) -> torch.Tensor:
    height, width = camera.height, camera.width
    x, y, z, indices = _project_world_to_image(means_world, camera)
    image = np.zeros((height, width, 3), dtype=np.float32)
    if indices.shape[0] == 0:
        return torch.from_numpy(image).permute(2, 0, 1)

    colors = values.detach().to(device=torch.device("cpu"), dtype=torch.float32).numpy()
    order = np.argsort(z)[::-1]
    for idx in order:
        image[y[idx], x[idx]] = colors[indices[idx]]
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def render_stored_normal_world_map(
    state: GaussianMapState,
    camera: PinholeCamera,
) -> torch.Tensor:
    if state.quats is None:
        raise ValueError("GaussianMapState must contain quats to render normals")
    normals = normals_from_quats(state.quats.to(dtype=torch.float32)) * 0.5 + 0.5
    rendered = _render_projected_attribute(
        values=normals.clamp(0.0, 1.0),
        means_world=state.means_world,
        camera=camera,
    )
    return F.normalize(rendered * 2.0 - 1.0, dim=0, eps=1e-6)


def as_chw_normal_map(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 4:
        value = value[0]
    if value.dim() == 3 and value.shape[-1] == 3:
        value = value.permute(2, 0, 1)
    elif value.dim() == 3 and value.shape[0] == 3:
        pass
    elif value.dim() == 2:
        value = value.unsqueeze(0).repeat(3, 1, 1)
    else:
        raise ValueError(f"Unsupported normal-map shape: {tuple(value.shape)}")
    return F.normalize(value, dim=0, eps=1e-6)


def _get_activated_splat_params(
    splats: torch.nn.ParameterDict,
) -> dict[str, torch.Tensor]:
    params = {
        "means": splats["means"],
        "quats": F.normalize(splats["quats"], dim=-1),
        "scales": torch.exp(splats["scales"]),
        "opacities": torch.sigmoid(splats["opacities"]).reshape(-1),
    }
    if "colors" in splats:
        params["colors"] = torch.sigmoid(splats["colors"])
        params["sh_degree"] = None
    else:
        params["colors"] = (
            torch.cat([splats["sh0"], splats["shN"]], dim=1)
            if "shN" in splats
            else splats["sh0"]
        )
        params["sh_degree"] = int(round(params["colors"].shape[1] ** 0.5) - 1)
    return params


def _safe_logit(values: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.logit(values.clamp(min=eps, max=1.0 - eps))


def _quats_from_normals(normals_world: torch.Tensor) -> torch.Tensor:
    normals_world = F.normalize(normals_world, dim=-1, eps=1e-6)
    z_axis = torch.tensor(
        [0.0, 0.0, 1.0], device=normals_world.device, dtype=normals_world.dtype
    ).expand_as(normals_world)
    dots = (z_axis * normals_world).sum(dim=-1, keepdim=True)
    xyz = torch.cross(z_axis, normals_world, dim=-1)
    quats = torch.cat([1.0 + dots, xyz], dim=-1)

    opposite = dots.squeeze(-1) < -0.9999
    if opposite.any():
        fallback = torch.zeros(
            (int(opposite.sum().item()), 4),
            device=normals_world.device,
            dtype=normals_world.dtype,
        )
        fallback[:, 2] = 1.0
        quats[opposite] = fallback

    small = quats.norm(dim=-1) < 1e-8
    if small.any():
        quats[small, 0] = 1.0
        quats[small, 1:] = 0.0
    return F.normalize(quats, dim=-1, eps=1e-6)


def gaussian_map_to_splats(
    state: GaussianMapState,
    device: torch.device | str = "cpu",
    orient_to_normals: bool = False,
    thickness_scale: float = 1.0,
    planar_scale: float = 1.0,
) -> torch.nn.ParameterDict:
    device = torch.device(device)
    means = state.means_world.to(device=device, dtype=torch.float32)
    colors = state.colors.to(device=device, dtype=torch.float32)
    scales = state.scales.to(device=device, dtype=torch.float32).clamp_min(1e-6)
    if planar_scale != 1.0:
        scales = scales.clone()
        scales[:, :2] = (scales[:, :2] * planar_scale).clamp_min(1e-6)
    if thickness_scale != 1.0:
        scales = scales.clone()
        scales[:, 2] = (scales[:, :2].mean(dim=-1) * thickness_scale).clamp_min(1e-6)
    opacities = state.opacities.to(device=device, dtype=torch.float32).reshape(-1)

    if state.quats is not None:
        quats = state.quats.to(device=device, dtype=torch.float32)
        if quats.shape != (means.shape[0], 4):
            raise ValueError(
                f"GaussianMapState quats must have shape {(means.shape[0], 4)}, "
                f"got {tuple(quats.shape)}"
            )
        quats = F.normalize(quats, dim=-1, eps=1e-6)
    else:
        quats = torch.zeros((means.shape[0], 4), device=device, dtype=torch.float32)
        quats[:, 0] = 1.0

    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means.clone()),
            "quats": torch.nn.Parameter(quats),
            "scales": torch.nn.Parameter(torch.log(scales)),
            "opacities": torch.nn.Parameter(_safe_logit(opacities)),
            "colors": torch.nn.Parameter(_safe_logit(colors)),
        }
    )


def render_material_channels(
    splats: torch.nn.ParameterDict,
    material_state: GaussianMaterialState,
    camera: PinholeCamera,
    render_types: tuple[str, ...] = ("albedo", "roughness", "metallic"),
    gsplat_root: str | Path | None = None,
    device: torch.device | str = "cuda",
) -> dict[str, torch.Tensor]:
    ensure_local_gsplat_path(gsplat_root)
    from gsplat.rendering import rasterization

    device = torch.device(device)
    activated = _get_activated_splat_params(splats)
    means = activated["means"].to(device=device)
    quats = activated["quats"].to(device=device)
    scales = activated["scales"].to(device=device)
    opacities = activated["opacities"].to(device=device)

    mat_state = material_state.to(device)
    camera = camera.to(device)
    viewmats = _make_viewmat(camera, device, means.dtype).unsqueeze(0)
    Ks = _make_intrinsics(camera, device, means.dtype).unsqueeze(0)

    rendered: dict[str, torch.Tensor] = {}
    for render_type in render_types:
        if render_type == "albedo":
            colors = mat_state.albedo
        elif render_type == "roughness":
            colors = mat_state.roughness.repeat(1, 3)
        elif render_type == "metallic":
            colors = mat_state.metallic.repeat(1, 3)
        else:
            raise KeyError(f"Unsupported render_type: {render_type}")

        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=camera.width,
            height=camera.height,
            sh_degree=None,
            packed=False,
            render_mode="RGB",
        )
        rendered[render_type] = render_colors[0].permute(2, 0, 1).clamp(0.0, 1.0)
    return rendered


def render_gaussian_map_channels(
    state: GaussianMapState,
    camera: PinholeCamera,
    render_types: tuple[str, ...] = ("rgb", "confidence"),
    gsplat_root: str | Path | None = None,
    device: torch.device | str = "cuda",
    backend: str = "auto",
    allow_fallback: bool = True,
    planar_scale: float | None = None,
    thickness_scale: float | None = None,
) -> dict[str, torch.Tensor]:
    if backend not in {"auto", "gsplat", "gsplat_2dgs", "simple"}:
        raise ValueError(f"Unsupported preview backend: {backend}")
    special_2dgs_render_types = {"gs_render_normal", "gs_surf_normal"}
    requested_special_2dgs = tuple(
        x for x in render_types if x in special_2dgs_render_types
    )
    if requested_special_2dgs and backend != "gsplat_2dgs":
        raise ValueError(
            f"{', '.join(requested_special_2dgs)} requires --preview_backend gsplat_2dgs"
        )

    def posterior_values(
        render_type: str, *, value_device, value_dtype
    ) -> torch.Tensor | None:
        value = getattr(state, render_type, None)
        if value is None:
            return None
        value = torch.nan_to_num(
            value.to(device=value_device, dtype=value_dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if render_type.endswith("_support") or render_type == "update_count":
            # A fixed log scale keeps support comparable across scenes and runs.
            value = torch.log1p(value) / torch.log(value.new_tensor(33.0))
        value = value.clamp(0.0, 1.0)
        if value.shape[-1] == 1:
            value = value.repeat(1, 3)
        return value

    posterior_render_types = {
        "albedo_variance",
        "roughness_variance",
        "metallic_variance",
        "albedo_support",
        "roughness_support",
        "metallic_support",
        "update_count",
    }

    if backend == "simple":
        confidence = state.confidence_sum.to(dtype=torch.float32)
        confidence = confidence / confidence.max().clamp_min(1e-6)

        rendered = {}
        for render_type in render_types:
            if render_type == "rgb":
                values = state.colors.to(dtype=torch.float32)
            elif render_type == "albedo":
                values = state.albedo.to(dtype=torch.float32)
            elif render_type == "normal":
                if state.quats is None:
                    raise ValueError("GaussianMapState must contain quats to render normals")
                values = normals_from_quats(state.quats.to(dtype=torch.float32)) * 0.5 + 0.5
            elif render_type == "roughness":
                values = state.roughness.to(dtype=torch.float32).repeat(1, 3)
            elif render_type == "metallic":
                values = state.metallic.to(dtype=torch.float32).repeat(1, 3)
            elif render_type == "confidence":
                values = confidence.repeat(1, 3)
            elif render_type in posterior_render_types:
                values = posterior_values(
                    render_type,
                    value_device=state.means_world.device,
                    value_dtype=torch.float32,
                )
                if values is None:
                    values = torch.zeros_like(state.colors, dtype=torch.float32)
            else:
                raise KeyError(f"Unsupported render_type: {render_type}")
            rendered[render_type] = _render_projected_attribute(
                values=values.clamp(0.0, 1.0),
                means_world=state.means_world,
                camera=camera,
            ).clamp(0.0, 1.0)
        return rendered

    try:
        ensure_local_gsplat_path(gsplat_root)
        if backend == "gsplat_2dgs":
            from gsplat.rendering import rasterization_2dgs
        else:
            from gsplat.rendering import rasterization

        device = torch.device(device)
        splats = gaussian_map_to_splats(
            state,
            device=device,
            orient_to_normals=(backend == "gsplat_2dgs"),
            planar_scale=(
                max(float(planar_scale), 1e-6)
                if planar_scale is not None
                else (1.8 if backend == "gsplat_2dgs" else 1.0)
            ),
            thickness_scale=(
                max(float(thickness_scale), 1e-6)
                if thickness_scale is not None
                else (0.05 if backend == "gsplat_2dgs" else 1.0)
            ),
        )
        activated = _get_activated_splat_params(splats)
        means = activated["means"]
        quats = activated["quats"]
        scales = activated["scales"]
        opacities = activated["opacities"]

        camera = camera.to(device)
        viewmats = _make_viewmat(camera, device, means.dtype).unsqueeze(0)
        Ks = _make_intrinsics(camera, device, means.dtype).unsqueeze(0)

        confidence = state.confidence_sum.to(device=device, dtype=means.dtype)
        confidence = confidence / confidence.max().clamp_min(1e-6)

        rendered: dict[str, torch.Tensor] = {}
        if requested_special_2dgs:
            render_colors, _, render_normals, surf_normals, _, _, _ = (
                rasterization_2dgs(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=state.albedo.to(device=device, dtype=means.dtype),
                    viewmats=viewmats,
                    Ks=Ks,
                    width=camera.width,
                    height=camera.height,
                    sh_degree=None,
                    packed=False,
                    render_mode="RGB+ED",
                )
            )
            del render_colors
            world_to_camera_rot = camera.world_to_camera[:3, :3].to(
                device=device, dtype=means.dtype
            )
            if "gs_render_normal" in requested_special_2dgs:
                render_normals = as_chw_normal_map(render_normals)
                render_normals = F.normalize(
                    torch.einsum("ij,jhw->ihw", world_to_camera_rot, render_normals),
                    dim=0,
                    eps=1e-6,
                )
                rendered["gs_render_normal"] = (render_normals * 0.5 + 0.5).clamp(
                    0.0, 1.0
                )
            if "gs_surf_normal" in requested_special_2dgs:
                surf_normals = as_chw_normal_map(surf_normals)
                surf_normals = F.normalize(
                    torch.einsum("ij,jhw->ihw", world_to_camera_rot, surf_normals),
                    dim=0,
                    eps=1e-6,
                )
                rendered["gs_surf_normal"] = (surf_normals * 0.5 + 0.5).clamp(0.0, 1.0)
        for render_type in render_types:
            if render_type in special_2dgs_render_types:
                continue
            if render_type == "rgb":
                colors = activated["colors"]
            elif render_type == "albedo":
                colors = state.albedo.to(device=device, dtype=means.dtype)
            elif render_type == "normal":
                if state.quats is None:
                    raise ValueError("GaussianMapState must contain quats to render normals")
                colors = (
                    normals_from_quats(state.quats.to(device=device, dtype=means.dtype))
                    * 0.5
                    + 0.5
                )
            elif render_type == "roughness":
                colors = state.roughness.to(device=device, dtype=means.dtype).repeat(
                    1, 3
                )
            elif render_type == "metallic":
                colors = state.metallic.to(device=device, dtype=means.dtype).repeat(
                    1, 3
                )
            elif render_type == "confidence":
                colors = confidence.repeat(1, 3)
            elif render_type in posterior_render_types:
                colors = posterior_values(
                    render_type,
                    value_device=device,
                    value_dtype=means.dtype,
                )
                if colors is None:
                    colors = torch.zeros_like(
                        state.colors, device=device, dtype=means.dtype
                    )
            else:
                raise KeyError(f"Unsupported render_type: {render_type}")

            if backend == "gsplat_2dgs":
                render_colors, _, _, _, _, _, _ = rasterization_2dgs(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=camera.width,
                    height=camera.height,
                    sh_degree=None,
                    packed=False,
                    render_mode="RGB",
                )
            else:
                render_colors, _, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=camera.width,
                    height=camera.height,
                    sh_degree=None,
                    packed=False,
                    render_mode="RGB",
                )
            rendered[render_type] = render_colors[0].permute(2, 0, 1).clamp(0.0, 1.0)
        return rendered
    except Exception:
        if not allow_fallback or backend == "gsplat":
            raise
        confidence = state.confidence_sum.to(dtype=torch.float32)
        confidence = confidence / confidence.max().clamp_min(1e-6)

        rendered = {}
        for render_type in render_types:
            if render_type in special_2dgs_render_types:
                raise ValueError(
                    f"{render_type} requires gsplat_2dgs preview rendering"
                )
            if render_type == "rgb":
                values = state.colors.to(dtype=torch.float32)
            elif render_type == "albedo":
                values = state.albedo.to(dtype=torch.float32)
            elif render_type == "normal":
                if state.quats is None:
                    raise ValueError("GaussianMapState must contain quats to render normals")
                values = normals_from_quats(state.quats.to(dtype=torch.float32)) * 0.5 + 0.5
            elif render_type == "roughness":
                values = state.roughness.to(dtype=torch.float32).repeat(1, 3)
            elif render_type == "metallic":
                values = state.metallic.to(dtype=torch.float32).repeat(1, 3)
            elif render_type == "confidence":
                values = confidence.repeat(1, 3)
            elif render_type in posterior_render_types:
                values = posterior_values(
                    render_type,
                    value_device=state.means_world.device,
                    value_dtype=torch.float32,
                )
                if values is None:
                    values = torch.zeros_like(state.colors, dtype=torch.float32)
            else:
                raise KeyError(f"Unsupported render_type: {render_type}")
            rendered[render_type] = _render_projected_attribute(
                values=values.clamp(0.0, 1.0),
                means_world=state.means_world,
                camera=camera,
            ).clamp(0.0, 1.0)
        return rendered


def render_gaussian_map_association(
    state: GaussianMapState,
    camera: PinholeCamera,
    gsplat_root: str | Path | None = None,
    device: torch.device | str = "cuda",
    backend: str = "gsplat_2dgs",
    planar_scale: float = 1.8,
    thickness_scale: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Render coverage and alpha-normalized material maps."""
    if backend != "gsplat_2dgs":
        raise ValueError(
            "Combined gaussian association rendering requires backend='gsplat_2dgs'"
        )

    ensure_local_gsplat_path(gsplat_root)
    from gsplat.rendering import rasterization_2dgs

    device = torch.device(device)
    with torch.no_grad():
        splats = gaussian_map_to_splats(
            state,
            device=device,
            orient_to_normals=True,
            planar_scale=max(float(planar_scale), 1e-6),
            thickness_scale=max(float(thickness_scale), 1e-6),
        )
        activated = _get_activated_splat_params(splats)
        means = activated["means"]
        features = torch.cat(
            (
                state.albedo.to(device=device, dtype=means.dtype),
                state.roughness.to(device=device, dtype=means.dtype),
                state.metallic.to(device=device, dtype=means.dtype),
            ),
            dim=1,
        )

        camera = camera.to(device)
        viewmats = _make_viewmat(camera, device, means.dtype).unsqueeze(0)
        Ks = _make_intrinsics(camera, device, means.dtype).unsqueeze(0)
        render_features, render_alphas, _, _, _, _, _ = rasterization_2dgs(
            means=means,
            quats=activated["quats"],
            scales=activated["scales"],
            opacities=activated["opacities"],
            colors=features,
            viewmats=viewmats,
            Ks=Ks,
            width=camera.width,
            height=camera.height,
            sh_degree=None,
            packed=False,
            render_mode="RGB",
        )

        coverage = render_alphas[0, ..., 0].clamp(0.0, 1.0)
        rendered = render_features[0] / coverage.clamp_min(1e-6).unsqueeze(-1)
        rendered = torch.where(
            (coverage > 1e-4).unsqueeze(-1),
            rendered,
            torch.zeros_like(rendered),
        )
        return {
            "coverage": coverage.detach(),
            "albedo": rendered[..., :3].permute(2, 0, 1).clamp(0.0, 1.0).detach(),
            "roughness": rendered[..., 3:4].permute(2, 0, 1).clamp(0.0, 1.0).detach(),
            "metallic": rendered[..., 4:5].permute(2, 0, 1).clamp(0.0, 1.0).detach(),
        }


def render_gaussian_map_coverage(
    state: GaussianMapState,
    camera: PinholeCamera,
    gsplat_root: str | Path | None = None,
    device: torch.device | str = "cuda",
    backend: str = "gsplat_2dgs",
) -> torch.Tensor:
    if backend not in {"gsplat", "gsplat_2dgs"}:
        raise ValueError(f"Unsupported coverage backend: {backend}")

    ensure_local_gsplat_path(gsplat_root)
    if backend == "gsplat_2dgs":
        from gsplat.rendering import rasterization_2dgs
    else:
        from gsplat.rendering import rasterization

    device = torch.device(device)
    splats = gaussian_map_to_splats(
        state,
        device=device,
        orient_to_normals=(backend == "gsplat_2dgs"),
        planar_scale=1.8 if backend == "gsplat_2dgs" else 1.0,
        thickness_scale=0.05 if backend == "gsplat_2dgs" else 1.0,
    )
    activated = _get_activated_splat_params(splats)
    means = activated["means"]
    quats = activated["quats"]
    scales = activated["scales"]
    opacities = activated["opacities"]

    camera = camera.to(device)
    viewmats = _make_viewmat(camera, device, means.dtype).unsqueeze(0)
    Ks = _make_intrinsics(camera, device, means.dtype).unsqueeze(0)
    colors = torch.ones((means.shape[0], 3), device=device, dtype=means.dtype)

    if backend == "gsplat_2dgs":
        _, render_alphas, _, _, _, _, _ = rasterization_2dgs(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=camera.width,
            height=camera.height,
            sh_degree=None,
            packed=False,
            render_mode="RGB",
        )
    else:
        _, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=camera.width,
            height=camera.height,
            sh_degree=None,
            packed=False,
            render_mode="RGB",
        )
    return render_alphas[0, ..., 0].clamp(0.0, 1.0)


def load_camera_dict(
    path: str | Path, pose_override_path: str | Path | None = None
) -> dict[str, PinholeCamera]:
    return load_camera_manifest(path, pose_override_path=pose_override_path)
