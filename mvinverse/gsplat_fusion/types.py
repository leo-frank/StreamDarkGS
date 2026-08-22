from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PinholeCamera:
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: torch.Tensor

    def __post_init__(self) -> None:
        self.camera_to_world = torch.as_tensor(self.camera_to_world, dtype=torch.float32)
        if self.camera_to_world.shape != (4, 4):
            raise ValueError(
                f"camera_to_world for {self.image_name} must have shape (4, 4), "
                f"got {tuple(self.camera_to_world.shape)}"
            )

    @property
    def world_to_camera(self) -> torch.Tensor:
        return torch.linalg.inv(self.camera_to_world)

    @property
    def rotation_camera_to_world(self) -> torch.Tensor:
        return self.camera_to_world[:3, :3]

    def scaled_to(self, width: int, height: int) -> "PinholeCamera":
        sx = float(width) / float(self.width)
        sy = float(height) / float(self.height)
        return PinholeCamera(
            image_name=self.image_name,
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            camera_to_world=self.camera_to_world.clone(),
        )

    def to(self, device: torch.device) -> "PinholeCamera":
        return PinholeCamera(
            image_name=self.image_name,
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            camera_to_world=self.camera_to_world.to(device=device),
        )


@dataclass
class MaterialMaps:
    albedo: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    normal: torch.Tensor
    shading: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    material_latent: torch.Tensor | None = None
    albedo_uncertainty: torch.Tensor | None = None
    roughness_uncertainty: torch.Tensor | None = None
    metallic_uncertainty: torch.Tensor | None = None
    normal_uncertainty: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name in ("albedo", "roughness", "metallic", "normal"):
            value = getattr(self, name)
            if value.dim() != 3:
                raise ValueError(f"{name} must have shape [C, H, W], got {tuple(value.shape)}")
        if self.albedo.shape[0] != 3:
            raise ValueError("albedo must have 3 channels")
        if self.roughness.shape[0] != 1:
            raise ValueError("roughness must have 1 channel")
        if self.metallic.shape[0] != 1:
            raise ValueError("metallic must have 1 channel")
        if self.normal.shape[0] != 3:
            raise ValueError("normal must have 3 channels")
        if self.shading is not None and self.shading.shape[0] != 3:
            raise ValueError("shading must have 3 channels")
        if self.confidence is not None and self.confidence.shape[0] != 1:
            raise ValueError("confidence must have 1 channel")
        for name in (
            "albedo_uncertainty",
            "roughness_uncertainty",
            "metallic_uncertainty",
            "normal_uncertainty",
        ):
            value = getattr(self, name)
            if value is not None and value.shape != (1, *self.image_hw):
                raise ValueError(f"{name} must have shape [1, H, W]")

    @property
    def image_hw(self) -> tuple[int, int]:
        return int(self.albedo.shape[-2]), int(self.albedo.shape[-1])

    def with_confidence(self) -> "MaterialMaps":
        if self.confidence is not None:
            return self
        height, width = self.image_hw
        confidence = torch.ones(
            (1, height, width),
            dtype=self.albedo.dtype,
            device=self.albedo.device,
        )
        return MaterialMaps(
            albedo=self.albedo,
            roughness=self.roughness,
            metallic=self.metallic,
            normal=self.normal,
            shading=self.shading,
            confidence=confidence,
            material_latent=self.material_latent,
            albedo_uncertainty=self.albedo_uncertainty,
            roughness_uncertainty=self.roughness_uncertainty,
            metallic_uncertainty=self.metallic_uncertainty,
            normal_uncertainty=self.normal_uncertainty,
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        payload = {
            "albedo": self.albedo,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "normal": self.normal,
        }
        for name in (
            "shading",
            "confidence",
            "material_latent",
            "albedo_uncertainty",
            "roughness_uncertainty",
            "metallic_uncertainty",
            "normal_uncertainty",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


@dataclass
class FrameMaterialEstimate:
    image_name: str
    camera: PinholeCamera
    materials: MaterialMaps
    depth: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.depth is not None:
            if self.depth.dim() != 3 or self.depth.shape[0] != 1:
                raise ValueError("depth must have shape [1, H, W]")


@dataclass
class GaussianMaterialState:
    means_world: torch.Tensor
    albedo: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    normal_world: torch.Tensor
    confidence_sum: torch.Tensor
    update_count: torch.Tensor
    material_latent: torch.Tensor | None = None

    @classmethod
    def initialize(
        cls,
        means_world: torch.Tensor,
        latent_dim: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "GaussianMaterialState":
        means_world = torch.as_tensor(means_world, dtype=dtype, device=device)
        if means_world.dim() != 2 or means_world.shape[1] != 3:
            raise ValueError(f"means_world must have shape [N, 3], got {tuple(means_world.shape)}")
        n = means_world.shape[0]
        material_latent = None
        if latent_dim is not None:
            material_latent = torch.zeros((n, latent_dim), dtype=dtype, device=means_world.device)
        return cls(
            means_world=means_world,
            albedo=torch.zeros((n, 3), dtype=dtype, device=means_world.device),
            roughness=torch.zeros((n, 1), dtype=dtype, device=means_world.device),
            metallic=torch.zeros((n, 1), dtype=dtype, device=means_world.device),
            normal_world=F.normalize(
                torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=means_world.device).repeat(n, 1),
                dim=-1,
            ),
            confidence_sum=torch.zeros((n, 1), dtype=dtype, device=means_world.device),
            update_count=torch.zeros((n, 1), dtype=dtype, device=means_world.device),
            material_latent=material_latent,
        )

    def to(self, device: torch.device) -> "GaussianMaterialState":
        return GaussianMaterialState(
            means_world=self.means_world.to(device=device),
            albedo=self.albedo.to(device=device),
            roughness=self.roughness.to(device=device),
            metallic=self.metallic.to(device=device),
            normal_world=self.normal_world.to(device=device),
            confidence_sum=self.confidence_sum.to(device=device),
            update_count=self.update_count.to(device=device),
            material_latent=None if self.material_latent is None else self.material_latent.to(device=device),
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        payload = {
            "means_world": self.means_world,
            "albedo": self.albedo,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "normal_world": self.normal_world,
            "confidence_sum": self.confidence_sum,
            "update_count": self.update_count,
        }
        if self.material_latent is not None:
            payload["material_latent"] = self.material_latent
        return payload
