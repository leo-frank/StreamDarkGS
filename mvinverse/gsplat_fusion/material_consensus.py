from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from .types import MaterialMaps

if TYPE_CHECKING:
    from .live_material_renderer import LiveMaterialRenderer


MATERIAL_CHANNELS = ("albedo", "roughness", "metallic", "normal")
MATERIAL_POLICIES = ("first", "latest", "robust_consensus")


def _resize_like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] == reference.shape[-2:]:
        return value
    return F.interpolate(
        value.unsqueeze(0),
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _huber_consensus(values: list[torch.Tensor], delta: float) -> torch.Tensor:
    reference = values[0]
    stack = torch.stack(
        [_resize_like(value, reference).float() for value in values], dim=0
    )
    if len(values) <= 2:
        return stack.mean(dim=0)
    center = stack.median(dim=0).values
    residual = (stack - center.unsqueeze(0)).abs()
    weights = (float(delta) / residual.clamp_min(1e-6)).clamp(max=1.0)
    return (stack * weights).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-6)


def _align_normal_axes(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    signs = value.new_tensor(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    candidates = value.unsqueeze(0) * signs[:, :, None, None]
    scores = (candidates * reference.unsqueeze(0)).sum(dim=1).mean(dim=(1, 2))
    return candidates[int(scores.argmax())]


class MVInverseMaterialStream:
    def __init__(
        self,
        *,
        ckpt: str = "",
        device: str = "cpu",
        max_long_edge: int = 512,
        policy: str = "robust_consensus",
        huber_delta: float = 0.1,
        renderer: "LiveMaterialRenderer | None" = None,
    ) -> None:
        if policy not in MATERIAL_POLICIES:
            raise ValueError(f"Unknown material overlap policy: {policy}")
        self.policy = policy
        self.huber_delta = max(float(huber_delta), 1e-6)
        if renderer is None:
            from .live_material_renderer import LiveMaterialRenderer

            renderer = LiveMaterialRenderer(
                ckpt=ckpt, device=device, max_long_edge=max_long_edge
            )
        self.renderer = renderer
        self._observations: dict[str, list[dict[str, torch.Tensor]]] = {}

    @staticmethod
    def _as_mapping(
        value: dict[str, torch.Tensor] | MaterialMaps,
    ) -> dict[str, torch.Tensor]:
        return value.as_dict() if isinstance(value, MaterialMaps) else value

    def add_window(
        self,
        outputs: dict[str, dict[str, torch.Tensor] | MaterialMaps],
    ) -> int:
        overlap_count = sum(stem in self._observations for stem in outputs)
        for stem, proposal in outputs.items():
            maps = self._as_mapping(proposal)
            missing = [key for key in MATERIAL_CHANNELS if key not in maps]
            if missing:
                raise KeyError(f"Material output for {stem} is missing {missing}")
            observation = {
                key: maps[key].detach().to(device="cpu") for key in MATERIAL_CHANNELS
            }
            self._observations.setdefault(stem, []).append(observation)
        return overlap_count

    def process_window(
        self,
        images: list[tuple[str, torch.Tensor]],
        *,
        target_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[dict[str, MaterialMaps], tuple[int, int], int]:
        proposals, input_size = self.renderer.propose_batch(
            images,
            output_device="cpu",
            target_size=target_size,
            output_size=output_size,
        )
        overlap_count = self.add_window(proposals)
        return proposals, input_size, overlap_count

    def pending_stems(self) -> set[str]:
        return set(self._observations)

    def resolve_frame(self, stem: str) -> tuple[dict[str, torch.Tensor], int]:
        observations = self._observations.get(stem)
        if not observations:
            raise KeyError(f"No MVInverse observations for {stem}")
        if self.policy == "first":
            return dict(observations[0]), len(observations)
        if self.policy == "latest":
            return dict(observations[-1]), len(observations)

        outputs = {
            key: _huber_consensus(
                [observation[key] for observation in observations],
                self.huber_delta,
            ).clamp(0.0, 1.0)
            for key in ("albedo", "roughness", "metallic")
        }
        reference = F.normalize(observations[0]["normal"].float(), dim=0, eps=1e-6)
        normals = [reference]
        for observation in observations[1:]:
            normal = F.normalize(
                _resize_like(observation["normal"], reference).float(),
                dim=0,
                eps=1e-6,
            )
            normals.append(_align_normal_axes(normal, reference))
        outputs["normal"] = F.normalize(
            torch.stack(normals).mean(dim=0), dim=0, eps=1e-6
        )
        return outputs, len(observations)

    def release_frame(self, stem: str) -> None:
        self._observations.pop(stem, None)

    def pop(self, stem: str) -> tuple[dict[str, torch.Tensor], int]:
        outputs = self.resolve_frame(stem)
        self.release_frame(stem)
        return outputs

    def pop_many(
        self,
        stems: list[str] | tuple[str, ...],
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, int]]:
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        counts: dict[str, int] = {}
        for stem in stems:
            if stem not in self._observations:
                continue
            outputs[stem], counts[stem] = self.pop(stem)
        return outputs, counts


MaterialObservationStream = MVInverseMaterialStream


def select_hybrid_material_window(
    image_names: list[str] | tuple[str, ...],
    window_start: int,
    window_stride: int,
    anchor_radius: int = 5,
    window_size: int = 5,
) -> tuple[str, ...]:
    del window_stride, anchor_radius
    start = max(int(window_start), 0)
    return tuple(image_names[start : start + max(int(window_size), 1)])
