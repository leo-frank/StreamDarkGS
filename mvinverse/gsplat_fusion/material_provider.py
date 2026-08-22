from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from .types import MaterialMaps


class MaterialProposalProvider(ABC):
    """Model-agnostic interface for 2D material proposal backbones."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def propose(self, image: torch.Tensor) -> MaterialMaps:
        """Predict one proposal from a CHW RGB image."""

    @abstractmethod
    def propose_batch(
        self,
        images: Sequence[tuple[str, torch.Tensor]],
        output_device: str | torch.device | None = None,
        target_size: tuple[int, int] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[dict[str, MaterialMaps], tuple[int, int]]:
        """Predict named proposals jointly from one material context."""


def proposal_to_legacy_dict(proposal: MaterialMaps) -> dict[str, torch.Tensor]:
    """Compatibility adapter for preview/export code that consumes mappings."""
    return proposal.as_dict()
