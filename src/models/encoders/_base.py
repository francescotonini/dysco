from typing import Dict

import torch
import torch.nn as nn

from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseEncoder(nn.Module):
    def __init__(self, *args, name: str, encoder_type: str, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )
        log.info(f"Loading {encoder_type} model: {name}")

        self._name = name.replace("/", "-").replace("\\", "-")
        self._encoder_type = encoder_type

        valid_encoders_type = ["text", "vision", "multimodal"]
        assert self._encoder_type in valid_encoders_type, (
            f"Invalid encoder type: {self._encoder_type}. "
            f"Must be one of: {valid_encoders_type}"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def encoder_type(self) -> str:
        return self._encoder_type

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_text_embeddings(self, *args, normalize=True, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_vision_embeddings(self, *args, normalize=True, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_multimodal_embeddings(
        self, *args, normalize=True, **kwargs
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def norm_if_needed(self, embeddings: torch.Tensor, normalize: bool) -> torch.Tensor:
        if normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def to_list(self, x):
        if not isinstance(x, list):
            x = [x]

        return x

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
