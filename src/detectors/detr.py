import pickle
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn


class DETR(nn.Module):
    def __init__(self, *args, cache_path: str, **kwargs):
        super().__init__()

        self._name = "detr"
        self._cache_path: Path = Path(cache_path)
        assert self._cache_path.exists(), f"Cache path {self._cache_path} does not exist."

        with open(self._cache_path, "rb") as f:
            self._data = pickle.load(f)

    @property
    def name(self) -> str:
        return self._name

    def forward(self, batch: Dict[str, Any]) -> Dict[str, List[torch.Tensor]]:
        device = batch["images_tensor"][0].device

        all_boxes = []
        all_scores = []
        all_labels = []
        for filename in batch["images_filename"]:
            data = self._data[filename][0]
            boxes = torch.tensor(data["boxes"], device=device, dtype=torch.float)
            scores = torch.tensor(data["scores"], device=device, dtype=torch.float)
            labels = torch.tensor(data["labels"], device=device, dtype=torch.long)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return {
            "boxes": all_boxes,
            "scores": all_scores,
            "labels": all_labels,
        }
