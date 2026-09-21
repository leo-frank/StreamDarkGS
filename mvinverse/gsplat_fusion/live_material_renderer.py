from __future__ import annotations

import contextlib
import importlib
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from mvinverse.gsplat_fusion.material_provider import MaterialProposalProvider
from mvinverse.gsplat_fusion.rgbd import depth_to_world_points, normals_from_depth
from mvinverse.gsplat_fusion.types import MaterialMaps
from mvinverse.models.mvinverse import MVInverse


def load_model(
    checkpoint: str,
    device: torch.device,
) -> MVInverse:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        print(f"Loading model from local Hugging Face snapshot: {checkpoint}")
        return MVInverse.from_pretrained(checkpoint).to(device).eval()

    if checkpoint_path.is_file():
        print(f"Loading model from local checkpoint: {checkpoint}")
        model = MVInverse().to(device).eval()
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        weights = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            print(f"Warning: missing checkpoint keys: {missing}")
        if unexpected:
            print(f"Warning: unexpected checkpoint keys: {unexpected}")
        return model

    print(f"Loading model from Hugging Face Hub: {checkpoint}")
    return MVInverse.from_pretrained(checkpoint).to(device).eval()


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


def align_normals_camera(
    source_normals_camera: torch.Tensor,
    target_normals_camera: torch.Tensor,
) -> torch.Tensor:
    source = F.normalize(source_normals_camera, dim=0, eps=1e-6)
    target = F.normalize(target_normals_camera, dim=0, eps=1e-6)
    valid = (source.norm(dim=0) > 1e-6) & (target.norm(dim=0) > 1e-6)
    if not valid.any():
        return source

    best_score = None
    best = source
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                candidate = source * source.new_tensor([sx, sy, sz]).view(3, 1, 1)
                score = (candidate * target).sum(dim=0)[valid].mean()
                if best_score is None or score > best_score:
                    best_score = score
                    best = candidate
    return F.normalize(best, dim=0, eps=1e-6)


MATERIAL_MAP_KEYS = ("albedo", "normal", "roughness", "metallic")
OPTIONAL_MATERIAL_MAP_KEYS = ("shading", "material_latent")


def _extract_frame_maps(
    predictions: dict[str, torch.Tensor],
    frame_index: int,
) -> dict[str, torch.Tensor]:
    maps = {
        key: predictions[key][0, frame_index].permute(2, 0, 1)
        for key in MATERIAL_MAP_KEYS
    }
    if "shading" in predictions:
        maps["shading"] = predictions["shading"][0, frame_index].permute(2, 0, 1)
    latent = predictions.get("material_latent")
    if latent is not None:
        maps["material_latent"] = latent[0, frame_index]
    return maps


def _finalize_frame_maps(
    maps: dict[str, torch.Tensor],
    output_size: tuple[int, int],
    output_device: torch.device,
) -> dict[str, torch.Tensor]:
    finalized: dict[str, torch.Tensor] = {}
    for key, value in maps.items():
        if key != "material_latent" and value.shape[-2:] != output_size:
            value = F.interpolate(
                value.unsqueeze(0),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if key == "normal":
            value = F.normalize(value, dim=0, eps=1e-6)
        elif key != "material_latent":
            value = value.clamp(0.0, 1.0)
        finalized[key] = value.detach().to(device=output_device)
    return finalized


class LiveMaterialRenderer(MaterialProposalProvider):
    def __init__(
        self,
        ckpt: str,
        device: str,
        max_long_edge: int,
        capture_material_latent: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.max_long_edge = max_long_edge
        self.capture_material_latent = bool(capture_material_latent)
        self.model = load_model(ckpt, self.device)

    @property
    def name(self) -> str:
        return "mvinverse"

    def _compute_model_size(self, height: int, width: int) -> tuple[int, int]:
        if max(width, height) > self.max_long_edge:
            scaling_factor = float(self.max_long_edge) / float(max(width, height))
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
        else:
            new_width, new_height = width, height

        new_w = max(new_width // 14 * 14, 14)
        new_h = max(new_height // 14 * 14, 14)
        return new_h, new_w

    def _resize_tensor_for_model(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 3:
            raise ValueError(
                f"Expected CHW image tensor, got shape {tuple(image.shape)}"
            )
        _, height, width = image.shape
        new_h, new_w = self._compute_model_size(height, width)
        if (new_h, new_w) == (height, width):
            return image.to(device=self.device, dtype=torch.float32, non_blocking=True)
        return F.interpolate(
            image.unsqueeze(0).to(
                device=self.device, dtype=torch.float32, non_blocking=True
            ),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def _run_model(self, model_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        autocast_context = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.no_grad():
            with autocast_context:
                return self.model(
                    model_inputs.unsqueeze(0),
                    return_material_latent=self.capture_material_latent,
                )

    def _resize_tensor_batch_for_model(
        self,
        images: list[torch.Tensor],
        target_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if not images:
            raise ValueError("images must not be empty")
        first = images[0]
        if first.dim() != 3:
            raise ValueError(
                f"Expected CHW image tensor, got shape {tuple(first.shape)}"
            )
        _, height, width = first.shape
        if target_size is None:
            new_h, new_w = self._compute_model_size(height, width)
        else:
            new_h, new_w = target_size
        batch = torch.stack(
            [
                image.to(device=self.device, dtype=torch.float32, non_blocking=True)
                for image in images
            ],
            dim=0,
        )
        if (new_h, new_w) == (height, width):
            return batch, (new_h, new_w)
        return (
            F.interpolate(
                batch,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ),
            (new_h, new_w),
        )

    def render_maps(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        model_input = self._resize_tensor_for_model(image)
        predictions = self._run_model(model_input.unsqueeze(0))
        return _finalize_frame_maps(
            _extract_frame_maps(predictions, 0),
            output_size=image.shape[-2:],
            output_device=torch.device("cpu"),
        )

    def propose(self, image: torch.Tensor) -> MaterialMaps:
        return MaterialMaps(**self.render_maps(image))

    def render_maps_batch(
        self,
        images: list[tuple[str, torch.Tensor]],
        output_device: str | torch.device | None = None,
        target_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], tuple[int, int]]:
        if not images:
            return {}, (0, 0)
        from .pipeline_profiling import StageClock
        clock = StageClock(self.device)
        final_size = output_size or images[0][1].shape[-2:]
        model_inputs, model_input_size = self._resize_tensor_batch_for_model(
            [image for _, image in images],
            target_size=target_size,
        )
        clock.mark("preprocess_and_upload")
        predictions = self._run_model(model_inputs)
        clock.mark("model")
        target_device = torch.device(output_device or "cpu")
        outputs_by_name = {
            name: _finalize_frame_maps(
                _extract_frame_maps(predictions, frame_index),
                output_size=final_size,
                output_device=target_device,
            )
            for frame_index, (name, _image) in enumerate(images)
        }
        clock.mark("resize_and_output_transfer")
        clock.report("mvinverse", first=images[0][0], frames=len(images))
        return outputs_by_name, model_input_size

    def propose_batch(
        self,
        images: list[tuple[str, torch.Tensor]],
        output_device: str | torch.device | None = None,
        target_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[dict[str, MaterialMaps], tuple[int, int]]:
        outputs, size = self.render_maps_batch(
            images,
            output_device=output_device,
            target_size=target_size,
            output_size=output_size,
        )
        material_keys = MATERIAL_MAP_KEYS + OPTIONAL_MATERIAL_MAP_KEYS
        return {
            name: MaterialMaps(
                **{key: maps[key] for key in material_keys if key in maps}
            )
            for name, maps in outputs.items()
        }, size

    def render_albedo(self, image: torch.Tensor) -> torch.Tensor:
        return self.render_maps(image)["albedo"]

    def render_relight(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        camera,
        light_dir_camera: torch.Tensor,
        light_color: torch.Tensor,
        relight_mode: str,
        flash_intensity: float,
        flash_radius: float,
        flash_beam_power: float,
        apply_tonemap: bool,
        specular_scale: float,
        ambient: float,
        normal_source: str,
        flip_normals_to_view: bool = False,
    ) -> torch.Tensor:
        maps = self.render_maps(image)
        albedo = maps["albedo"].to(dtype=image.dtype)
        roughness = maps["roughness"].to(dtype=image.dtype).clamp(0.02, 1.0)
        metallic = maps["metallic"].to(dtype=image.dtype).clamp(0.0, 1.0)
        points_world, _, _ = depth_to_world_points(depth, camera, stride=1)
        camera_center = camera.camera_to_world[:3, 3].to(
            device=image.device, dtype=image.dtype
        )
        viewdirs = F.normalize(
            camera_center.view(1, 1, 3) - points_world, dim=-1, eps=1e-6
        )
        depth_normals_world = (
            normals_from_depth(depth, camera).permute(2, 0, 1).to(dtype=image.dtype)
        )
        if normal_source == "mvinverse":
            depth_normals_camera = torch.einsum(
                "ij,jhw->ihw",
                camera.world_to_camera[:3, :3].to(
                    device=image.device, dtype=image.dtype
                ),
                depth_normals_world,
            )
            aligned_mvinverse_camera = align_normals_camera(
                maps["normal"].to(device=image.device, dtype=image.dtype),
                depth_normals_camera,
            )
            # aligned_mvinverse_camera[2:] = -aligned_mvinverse_camera[2:]
            # aligned_mvinverse_camera[1:] = -aligned_mvinverse_camera[1:]
            normals = torch.einsum(
                "ij,jhw->ihw",
                camera.rotation_camera_to_world.to(
                    device=image.device, dtype=image.dtype
                ),
                aligned_mvinverse_camera,
            )
        else:
            normals = depth_normals_world
        normal_view_dot = (normals.permute(1, 2, 0) * viewdirs).sum(
            dim=-1, keepdim=True
        )
        flip = torch.where(normal_view_dot < 0.0, -1.0, 1.0)
        normals = F.normalize(normals * flip.permute(2, 0, 1), dim=0, eps=1e-6)
        light_color = light_color.to(device=image.device, dtype=image.dtype)
        light_dir_camera = F.normalize(
            light_dir_camera.to(device=image.device, dtype=image.dtype),
            dim=0,
            eps=1e-6,
        )

        if relight_mode == "directional":
            light_dir_world = (
                (
                    camera.rotation_camera_to_world.to(
                        device=image.device, dtype=image.dtype
                    )
                    @ light_dir_camera
                )
                .view(1, 1, 3)
                .expand_as(points_world)
            )
            attenuation = torch.ones(
                (*points_world.shape[:2], 1),
                device=image.device,
                dtype=image.dtype,
            )
        else:
            flash_position = camera_center
            light_vec_world = flash_position.view(1, 1, 3) - points_world
            light_dir_world = F.normalize(light_vec_world, dim=-1, eps=1e-6)
            camera_forward = F.normalize(
                camera.rotation_camera_to_world.to(
                    device=image.device, dtype=image.dtype
                )
                @ torch.tensor([0.0, 0.0, 1.0], device=image.device, dtype=image.dtype),
                dim=0,
                eps=1e-6,
            )
            beam_dir = F.normalize(
                points_world - camera_center.view(1, 1, 3), dim=-1, eps=1e-6
            )
            beam = (
                (beam_dir * camera_forward.view(1, 1, 3))
                .sum(dim=-1, keepdim=True)
                .clamp_min(0.0)
            )
            beam = beam.pow(max(float(flash_beam_power), 1.0))
            flash_radius = max(float(flash_radius), 1e-3)
            dist2 = light_vec_world.square().sum(dim=-1, keepdim=True)
            attenuation = (
                float(flash_intensity)
                * beam
                / (1.0 + dist2 / (flash_radius * flash_radius))
            )

        return _pbr_relight(
            albedo=albedo,
            roughness=roughness,
            metallic=metallic,
            normals=normals,
            viewdirs=viewdirs.permute(2, 0, 1),
            light_dir=light_dir_world,
            light_color=light_color,
            light_attenuation=attenuation.permute(2, 0, 1),
            ambient=ambient,
            apply_tonemap=apply_tonemap,
            specular_scale=specular_scale,
            flip_normals_to_view=flip_normals_to_view,
        )


def _pbr_relight(
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    normals: torch.Tensor,
    viewdirs: torch.Tensor,
    light_dir: torch.Tensor,
    light_color: torch.Tensor,
    light_attenuation: torch.Tensor,
    ambient: float,
    apply_tonemap: bool,
    specular_scale: float,
    flip_normals_to_view: bool = False,
    ambient_irradiance: torch.Tensor | None = None,
    ambient_occlusion: torch.Tensor | None = None,
    ao_direct_strength: float = 0.0,
    extra_lights: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] | None = None,
    output_encoding: str = "srgb",
) -> torch.Tensor:
    albedo_hw3 = albedo.permute(1, 2, 0)
    normals_hw3 = F.normalize(normals.permute(1, 2, 0), dim=-1, eps=1e-6)
    viewdirs_hw3 = F.normalize(viewdirs.permute(1, 2, 0), dim=-1, eps=1e-6)
    if flip_normals_to_view:
        normal_view_dot = (normals_hw3 * viewdirs_hw3).sum(dim=-1, keepdim=True)
        normals_hw3 = torch.where(
            normal_view_dot < 0.0,
            -normals_hw3,
            normals_hw3,
        )
    rough_hw1 = roughness.permute(1, 2, 0).clamp(0.02, 1.0)
    metal_hw1 = metallic.permute(1, 2, 0).clamp(0.0, 1.0)
    light_hw3 = F.normalize(light_dir, dim=-1, eps=1e-6)
    light_color_hw3 = light_color.view(1, 1, 3)
    atten_hw1 = light_attenuation.permute(1, 2, 0).clamp_min(0.0)

    half_dirs = F.normalize(light_hw3 + viewdirs_hw3, dim=-1, eps=1e-6)
    n_dot_l = (normals_hw3 * light_hw3).sum(dim=-1, keepdim=True).clamp_min(0.0)
    n_dot_v = (normals_hw3 * viewdirs_hw3).sum(dim=-1, keepdim=True).clamp_min(0.0)
    n_dot_h = (normals_hw3 * half_dirs).sum(dim=-1, keepdim=True).clamp_min(0.0)
    h_dot_v = (half_dirs * viewdirs_hw3).sum(dim=-1, keepdim=True).clamp_min(0.0)

    f0 = 0.04 * (1.0 - metal_hw1) + albedo_hw3 * metal_hw1
    fresnel = f0 + (1.0 - f0) * (1.0 - h_dot_v).pow(5)
    diffuse = (1.0 - fresnel) * (1.0 - metal_hw1) * albedo_hw3 / math.pi

    alpha = rough_hw1.pow(2.0).clamp_min(1e-4)
    alpha2 = alpha.pow(2.0)
    denom = (n_dot_h.pow(2.0) * (alpha2 - 1.0) + 1.0).pow(2.0).clamp_min(1e-4)
    D = alpha2 / (math.pi * denom)
    k = (rough_hw1 + 1.0).pow(2.0) / 8.0
    G_l = n_dot_l / (n_dot_l * (1.0 - k) + k).clamp_min(1e-4)
    G_v = n_dot_v / (n_dot_v * (1.0 - k) + k).clamp_min(1e-4)
    specular = (D * fresnel * G_l * G_v) / (4.0 * n_dot_l * n_dot_v).clamp_min(1e-4)
    specular = specular * float(specular_scale)

    flash = light_color_hw3 * atten_hw1
    direct_linear = flash * (diffuse + specular) * n_dot_l
    for extra_direction, extra_color, extra_attenuation in extra_lights or ():
        extra_light_hw3 = F.normalize(extra_direction, dim=-1, eps=1e-6)
        extra_color_hw3 = extra_color.view(1, 1, 3)
        extra_atten_hw1 = extra_attenuation.permute(1, 2, 0).clamp_min(0.0)
        extra_half_dirs = F.normalize(
            extra_light_hw3 + viewdirs_hw3, dim=-1, eps=1e-6
        )
        extra_n_dot_l = (
            (normals_hw3 * extra_light_hw3)
            .sum(dim=-1, keepdim=True)
            .clamp_min(0.0)
        )
        extra_n_dot_h = (
            (normals_hw3 * extra_half_dirs)
            .sum(dim=-1, keepdim=True)
            .clamp_min(0.0)
        )
        extra_h_dot_v = (
            (extra_half_dirs * viewdirs_hw3)
            .sum(dim=-1, keepdim=True)
            .clamp_min(0.0)
        )
        extra_fresnel = f0 + (1.0 - f0) * (1.0 - extra_h_dot_v).pow(5)
        extra_diffuse = (
            (1.0 - extra_fresnel)
            * (1.0 - metal_hw1)
            * albedo_hw3
            / math.pi
        )
        extra_denom = (
            extra_n_dot_h.pow(2.0) * (alpha2 - 1.0) + 1.0
        ).pow(2.0).clamp_min(1e-4)
        extra_D = alpha2 / (math.pi * extra_denom)
        extra_G_l = extra_n_dot_l / (
            extra_n_dot_l * (1.0 - k) + k
        ).clamp_min(1e-4)
        extra_specular = (
            extra_D * extra_fresnel * extra_G_l * G_v
        ) / (4.0 * extra_n_dot_l * n_dot_v).clamp_min(1e-4)
        extra_specular = extra_specular * float(specular_scale)
        direct_linear = direct_linear + (
            extra_color_hw3
            * extra_atten_hw1
            * (extra_diffuse + extra_specular)
            * extra_n_dot_l
        )
    if ambient_irradiance is None:
        ambient_hw3 = float(ambient) * torch.ones_like(albedo_hw3)
    else:
        ambient_hw3 = ambient_irradiance.to(
            device=albedo_hw3.device, dtype=albedo_hw3.dtype
        )
        if ambient_hw3.ndim == 3 and ambient_hw3.shape[0] == 3:
            ambient_hw3 = ambient_hw3.permute(1, 2, 0)
    if ambient_occlusion is None:
        ao_hw1 = torch.ones_like(n_dot_l)
    else:
        ao_hw1 = ambient_occlusion.to(
            device=albedo_hw3.device, dtype=albedo_hw3.dtype
        )
        if ao_hw1.ndim == 3 and ao_hw1.shape[0] == 1:
            ao_hw1 = ao_hw1.permute(1, 2, 0)
        ao_hw1 = ao_hw1.clamp(0.0, 1.0)
    direct_ao = 1.0 - min(max(float(ao_direct_strength), 0.0), 1.0) * (1.0 - ao_hw1)
    relit_linear = (
        ambient_hw3 * albedo_hw3 * ao_hw1
        + direct_linear * direct_ao
    )
    if apply_tonemap:
        relit_linear = relit_linear / (1.0 + relit_linear)
    else:
        relit_linear = relit_linear.clamp(0.0, 1.0)
    relit_linear = relit_linear.clamp(0.0, 1.0)
    if output_encoding == "linear":
        return relit_linear.permute(2, 0, 1)
    if output_encoding != "srgb":
        raise ValueError(f"Unknown relit output encoding: {output_encoding}")
    relit = torch.where(
        relit_linear <= 0.0031308,
        12.92 * relit_linear,
        1.055 * relit_linear.pow(1.0 / 2.4) - 0.055,
    )
    return relit.permute(2, 0, 1)


def _load_streaminverse_symbols(streaminverse_root: str):
    root = Path(streaminverse_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"StreamInverse root does not exist: {root}")
    if not (root / "online_inference.py").is_file():
        raise FileNotFoundError(
            f"StreamInverse online_inference.py not found under: {root}"
        )

    old_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "mvinverse" or name.startswith("mvinverse.")
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)

    try:
        sys.path.insert(0, str(root))
        online_inference = importlib.import_module("online_inference")
        stream_session = importlib.import_module("mvinverse.stream_session")
        return online_inference.load_model, stream_session.MVInverseStreamSession
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name == "mvinverse" or name.startswith("mvinverse.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = old_path


class StreamInverseMaterialRenderer(LiveMaterialRenderer):
    def __init__(
        self,
        streaminverse_root: str,
        ckpt: str,
        device: str,
        max_long_edge: int,
        mode: str = "full",
        window_size: int = 5,
        chunk_size: int = 3,
    ) -> None:
        self.streaminverse_root = str(Path(streaminverse_root).expanduser().resolve())
        self.streaminverse_mode = str(mode)
        self.streaminverse_window_size = max(int(window_size), 1)
        self.streaminverse_chunk_size = max(int(chunk_size), 1)
        load_streaminverse_model, stream_session_cls = _load_streaminverse_symbols(
            self.streaminverse_root
        )
        self._stream_session_cls = stream_session_cls
        self.device = torch.device(device)
        self.max_long_edge = max_long_edge
        self.capture_material_latent = False
        self.model = load_streaminverse_model(
            ckpt,
            self.device,
            mode=self.streaminverse_mode,
            window_size=self.streaminverse_window_size,
        )
        self._stream_session = None
        self._stream_image_store: dict[str, torch.Tensor] = {}
        self._stream_predictions_by_name: dict[str, dict[str, torch.Tensor]] = {}
        self._stream_order: list[str] = []
        self._stream_model_input_size: tuple[int, int] | None = None

    @property
    def name(self) -> str:
        return "streaminverse"

    @property
    def _uses_persistent_stream(self) -> bool:
        return self.streaminverse_mode in {"causal", "window", "chunk"}

    def _new_stream_session(self):
        return self._stream_session_cls(
            self.model,
            mode=self.streaminverse_mode,
            window_size=self.streaminverse_window_size,
        )

    def _clear_stream_state(self) -> None:
        self._stream_session = None
        self._stream_predictions_by_name.clear()
        self._stream_order.clear()

    def _store_stream_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        ordered_names: list[str],
        start_index: int = 0,
    ) -> None:
        for local_idx, name in enumerate(ordered_names):
            idx = start_index + local_idx
            if idx >= predictions["albedo"].shape[1]:
                break
            maps = _extract_frame_maps(predictions, idx)
            self._stream_predictions_by_name[name] = _finalize_frame_maps(
                maps,
                output_size=maps["albedo"].shape[-2:],
                output_device=torch.device("cpu"),
            )

    def _forward_stream_inputs(
        self,
        names: list[str],
        inputs_by_name: dict[str, torch.Tensor],
    ) -> None:
        if not names:
            return
        if self._stream_session is None:
            self._stream_session = self._new_stream_session()
        step = (
            self.streaminverse_chunk_size if self.streaminverse_mode == "chunk" else 1
        )
        for start in range(0, len(names), step):
            chunk_names = names[start : start + step]
            chunk = torch.stack(
                [
                    inputs_by_name[name].to(
                        device=self.device, dtype=torch.float32, non_blocking=True
                    )
                    for name in chunk_names
                ],
                dim=0,
            )
            result = self._stream_session.forward_stream(chunk)
            self._store_stream_predictions(
                result,
                chunk_names,
                start_index=0,
            )
            self._stream_order.extend(chunk_names)
            if hasattr(self._stream_session, "_clear_predictions"):
                self._stream_session._clear_predictions()
            del result, chunk
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _ensure_stream_predictions(
        self,
        model_inputs_by_name: dict[str, torch.Tensor],
        model_input_size: tuple[int, int],
    ) -> None:
        for name, tensor in model_inputs_by_name.items():
            self._stream_image_store[name] = tensor.detach().cpu()
        if self._stream_model_input_size is None:
            self._stream_model_input_size = model_input_size
        elif self._stream_model_input_size != model_input_size:
            self._stream_model_input_size = model_input_size
            self._clear_stream_state()

        ordered_all = sorted(self._stream_image_store, key=_natural_sort_key)
        if all(
            name in self._stream_predictions_by_name for name in model_inputs_by_name
        ):
            return

        if self._stream_order == ordered_all[: len(self._stream_order)]:
            new_names = ordered_all[len(self._stream_order) :]
            self._forward_stream_inputs(new_names, self._stream_image_store)
            return

        # A sparse material window can reveal an older anchor after newer frames.
        # Causal KV state cannot insert history, so replay the stored stream in order.
        self._clear_stream_state()
        self._forward_stream_inputs(ordered_all, self._stream_image_store)

    def render_maps_batch(
        self,
        images: list[tuple[str, torch.Tensor]],
        output_device: str | torch.device | None = None,
        target_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], tuple[int, int]]:
        if not self._uses_persistent_stream:
            return super().render_maps_batch(
                images,
                output_device=output_device,
                target_size=target_size,
                output_size=output_size,
            )
        if not images:
            return {}, (0, 0)
        final_size = output_size or images[0][1].shape[-2:]
        model_inputs, model_input_size = self._resize_tensor_batch_for_model(
            [image for _, image in images],
            target_size=target_size,
        )
        model_inputs_by_name = {
            name: model_inputs[idx].detach().cpu()
            for idx, (name, _image) in enumerate(images)
        }
        self._ensure_stream_predictions(model_inputs_by_name, model_input_size)
        target_device = torch.device(output_device or "cpu")
        missing = [
            name
            for name, _image in images
            if name not in self._stream_predictions_by_name
        ]
        if missing:
            raise KeyError(
                f"Missing StreamInverse predictions for: {', '.join(missing)}"
            )
        outputs_by_name = {
            name: _finalize_frame_maps(
                self._stream_predictions_by_name[name],
                output_size=final_size,
                output_device=target_device,
            )
            for name, _image in images
        }
        return outputs_by_name, model_input_size

    def _run_model(self, model_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        autocast_context = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.no_grad():
            with autocast_context:
                if self.streaminverse_mode in {"causal", "window", "chunk"}:
                    session = self._stream_session_cls(
                        self.model,
                        mode=self.streaminverse_mode,
                        window_size=self.streaminverse_window_size,
                    )
                    step = (
                        self.streaminverse_chunk_size
                        if self.streaminverse_mode == "chunk"
                        else 1
                    )
                    result: dict[str, torch.Tensor] | None = None
                    for start in range(0, model_inputs.shape[0], step):
                        end = min(start + step, model_inputs.shape[0])
                        result = session.forward_stream(model_inputs[start:end])
                    if result is None:
                        raise RuntimeError(
                            "StreamInverse received an empty model input batch"
                        )
                    return result
                return self.model(
                    imgs=model_inputs.unsqueeze(0), mode=self.streaminverse_mode
                )
