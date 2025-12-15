from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.models.encoders._base import BaseEncoder
from src.utils import img_ops
from src.utils.logger import get_logger

log = get_logger(__name__)


class CLIPEncoder(BaseEncoder):
    def __init__(
        self,
        *args,
        model_name: str = "openai/clip-vit-base-patch16",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            name=model_name,
            encoder_type="multimodal",
            **kwargs,
        )

        self._model: CLIPModel = CLIPModel.from_pretrained(model_name)
        self._processor: CLIPProcessor = CLIPProcessor.from_pretrained(model_name)
        self._logit_scale: torch.Tensor = self._model.logit_scale

    @property
    def logit_scale(self) -> torch.Tensor:
        return self._logit_scale

    @torch.no_grad()
    def get_multimodal_embeddings(
        self,
        images_pil: Union[List[Any], Any],
        texts: Union[List[str], str],
        normalize: bool = True,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        images_pil = self.to_list(images_pil)
        images_pil = [img_ops.expand2square(img) for img in images_pil]
        texts = self.to_list(texts)

        inputs = self._processor(
            text=texts,
            images=images_pil,
            return_tensors="pt",
            padding=True,
            max_length=77,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        images_embeddings = self._model.get_image_features(inputs["pixel_values"])
        images_embeddings = self.norm_if_needed(images_embeddings, normalize)

        texts_embeddings = self._model.get_text_features(inputs["input_ids"])
        texts_embeddings = self.norm_if_needed(texts_embeddings, normalize)

        return {"image_embeddings": images_embeddings, "text_embeddings": texts_embeddings}

    @torch.no_grad()
    def get_text_embeddings(
        self, texts: Union[List[str], str], normalize: bool = True, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        texts = self.to_list(texts)

        # Create a fake image to pass to the processor
        fake_image_pil = Image.fromarray(
            torch.zeros(3, 224, 224).numpy().astype(np.uint8).transpose(1, 2, 0)
        )
        fake_images_pil = [fake_image_pil] * len(texts)

        embeddings = self.get_multimodal_embeddings(
            images_pil=fake_images_pil, texts=texts, normalize=normalize
        )

        return {"text_embeddings": embeddings["text_embeddings"]}

    @torch.no_grad()
    def get_vision_embeddings(
        self, images_pil: Union[List[Any], Any], normalize: bool = True, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        images_pil = self.to_list(images_pil)
        images_pil = [img_ops.expand2square(img) for img in images_pil]

        inputs = self._processor(
            images=images_pil,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        images_embeddings = self._model.get_image_features(inputs["pixel_values"])
        images_embeddings = self.norm_if_needed(images_embeddings, normalize)

        return {"image_embeddings": images_embeddings}
