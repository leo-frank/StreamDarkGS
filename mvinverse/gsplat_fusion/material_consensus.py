from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from .types import MaterialMaps

if TYPE_CHECKING:
    from .live_material_renderer import LiveMaterialRenderer


MATERIAL_CHANNELS = ("albedo", "roughness", "metallic", "normal")
MATERIAL_ALIGN_CHANNELS = ("albedo", "roughness", "metallic")
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


def _resolve_material_observations(
    observations: list[dict[str, torch.Tensor]],
    *,
    policy: str,
    huber_delta: float,
) -> dict[str, torch.Tensor]:
    if policy == "first":
        return dict(observations[0])
    if policy == "latest":
        return dict(observations[-1])

    outputs = {
        key: _huber_consensus(
            [observation[key] for observation in observations],
            huber_delta,
        ).clamp(0.0, 1.0)
        for key in MATERIAL_ALIGN_CHANNELS
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
    outputs["normal"] = F.normalize(torch.stack(normals).mean(dim=0), dim=0, eps=1e-6)
    return outputs


def _estimate_window_log_offsets(
    outputs: dict[str, dict[str, torch.Tensor]],
    references: dict[str, dict[str, torch.Tensor]],
    *,
    min_pixels: int,
    max_log_offset: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    offsets: dict[str, torch.Tensor] = {}
    stats: dict[str, object] = {
        "applied": False,
        "overlap_frames": len(references),
        "valid_pixels": 0,
    }
    max_abs_offset = abs(float(max_log_offset))
    for channel in MATERIAL_ALIGN_CHANNELS:
        frame_offsets: list[torch.Tensor] = []
        channel_valid_pixels = 0
        for stem, reference_maps in references.items():
            current = outputs[stem][channel].float().clamp(1e-4, 1.0)
            reference = _resize_like(reference_maps[channel], current).float().clamp(
                1e-4, 1.0
            )
            valid = torch.isfinite(current).all(dim=0) & torch.isfinite(reference).all(
                dim=0
            )
            if channel == "albedo":
                valid &= (current.mean(dim=0) > 0.03) & (
                    reference.mean(dim=0) > 0.03
                )
            valid_pixels = int(valid.sum().item())
            channel_valid_pixels += valid_pixels
            if valid_pixels < int(min_pixels):
                continue
            difference = torch.log(reference) - torch.log(current)
            frame_offsets.append(difference[:, valid].median(dim=1).values)
        if frame_offsets:
            offset = torch.stack(frame_offsets, dim=0).median(dim=0).values.clamp(
                -max_abs_offset, max_abs_offset
            )
            offsets[channel] = offset
        stats[f"{channel}_valid_pixels"] = channel_valid_pixels
        stats["valid_pixels"] = int(stats["valid_pixels"]) + channel_valid_pixels
    stats["offsets"] = {
        key: [float(value) for value in offset.detach().cpu()]
        for key, offset in offsets.items()
    }
    stats["applied"] = bool(offsets)
    return offsets, stats


def _apply_window_log_offsets(
    outputs: dict[str, dict[str, torch.Tensor]],
    offsets: dict[str, torch.Tensor],
    *,
    strength: float,
) -> dict[str, dict[str, torch.Tensor]]:
    if not offsets or strength <= 0.0:
        return outputs
    aligned: dict[str, dict[str, torch.Tensor]] = {}
    for stem, maps in outputs.items():
        aligned_maps = dict(maps)
        for channel, offset in offsets.items():
            current = maps[channel].float().clamp(1e-4, 1.0)
            channel_offset = offset.to(current).view(-1, 1, 1) * float(strength)
            aligned_maps[channel] = torch.exp(torch.log(current) + channel_offset).clamp(
                0.0, 1.0
            ).to(maps[channel].dtype)
        aligned[stem] = aligned_maps
    return aligned


class MVInverseMaterialStream:
    def __init__(
        self,
        *,
        ckpt: str = "",
        device: str = "cpu",
        max_long_edge: int = 512,
        policy: str = "robust_consensus",
        huber_delta: float = 0.1,
        align_window_to_overlap: bool = True,
        window_align_strength: float = 1.0,
        window_align_max_log_offset: float = 0.35,
        window_align_min_pixels: int = 512,
        renderer: "LiveMaterialRenderer | None" = None,
    ) -> None:
        if policy not in MATERIAL_POLICIES:
            raise ValueError(f"Unknown material overlap policy: {policy}")
        self.policy = policy
        self.huber_delta = max(float(huber_delta), 1e-6)
        self.align_window_to_overlap = bool(align_window_to_overlap)
        self.window_align_strength = max(float(window_align_strength), 0.0)
        self.window_align_max_log_offset = max(float(window_align_max_log_offset), 0.0)
        self.window_align_min_pixels = max(int(window_align_min_pixels), 1)
        self.last_window_alignment_stats: dict[str, object] = {"applied": False}
        self.last_window_raw_outputs: dict[str, dict[str, torch.Tensor]] = {}
        self.last_window_aligned_outputs: dict[str, dict[str, torch.Tensor]] = {}
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
        normalized_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for stem, proposal in outputs.items():
            maps = self._as_mapping(proposal)
            missing = [key for key in MATERIAL_CHANNELS if key not in maps]
            if missing:
                raise KeyError(f"Material output for {stem} is missing {missing}")
            normalized_outputs[stem] = {
                key: maps[key].detach().to(device="cpu") for key in maps
            }
        self.last_window_raw_outputs = {
            stem: {key: value.clone() for key, value in maps.items()}
            for stem, maps in normalized_outputs.items()
        }

        references = {
            stem: _resolve_material_observations(
                self._observations[stem],
                policy=self.policy,
                huber_delta=self.huber_delta,
            )
            for stem in normalized_outputs
            if stem in self._observations
        }
        overlap_count = len(references)
        self.last_window_alignment_stats = {
            "applied": False,
            "overlap_frames": overlap_count,
            "valid_pixels": 0,
        }
        if self.align_window_to_overlap and references:
            offsets, stats = _estimate_window_log_offsets(
                normalized_outputs,
                references,
                min_pixels=self.window_align_min_pixels,
                max_log_offset=self.window_align_max_log_offset,
            )
            self.last_window_alignment_stats = stats
            normalized_outputs = _apply_window_log_offsets(
                normalized_outputs,
                offsets,
                strength=self.window_align_strength,
            )
        self.last_window_aligned_outputs = {
            stem: {key: value.clone() for key, value in maps.items()}
            for stem, maps in normalized_outputs.items()
        }

        for stem, maps in normalized_outputs.items():
            observation = {key: maps[key] for key in MATERIAL_CHANNELS}
            self._observations.setdefault(stem, []).append(observation)
        return overlap_count

    def process_window(
        self,
        images: list[tuple[str, torch.Tensor]],
        *,
        target_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[dict[str, MaterialMaps], tuple[int, int], int]:
        from .pipeline_profiling import StageClock
        clock = StageClock(self.renderer.device)
        proposals, input_size = self.renderer.propose_batch(
            images,
            output_device="cpu",
            target_size=target_size,
            output_size=output_size,
        )
        clock.mark("propose")
        overlap_count = self.add_window(proposals)
        clock.mark("window_alignment_and_cache")
        clock.report("material_window", frames=len(images))
        return proposals, input_size, overlap_count

    def pending_stems(self) -> set[str]:
        return set(self._observations)

    def resolve_frame(self, stem: str) -> tuple[dict[str, torch.Tensor], int]:
        observations = self._observations.get(stem)
        if not observations:
            raise KeyError(f"No MVInverse observations for {stem}")
        if self.policy == "first":
            return _resolve_material_observations(
                observations,
                policy=self.policy,
                huber_delta=self.huber_delta,
            ), len(observations)
        if self.policy == "latest":
            return _resolve_material_observations(
                observations,
                policy=self.policy,
                huber_delta=self.huber_delta,
            ), len(observations)
        return _resolve_material_observations(
            observations,
            policy=self.policy,
            huber_delta=self.huber_delta,
        ), len(observations)

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
