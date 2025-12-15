from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

from libs.long_clip.model import longclip
from src.models.encoders._base import BaseEncoder
from src.utils import img_ops
from src.utils.logger import get_logger

log = get_logger(__name__)


# TODO: de-hardcode the model paths
LONGCLIP_MODELS = {
    "longclip-B": "weights/longclip-B.pt",
    "longclip-L": "weights/longclip-L.pt",
}


class LongCLIPEncoder(BaseEncoder):
    def __init__(
        self,
        *args,
        model_name: str = "longclip-B",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            name=model_name,
            encoder_type="multimodal",
            **kwargs,
        )

        model_path = LONGCLIP_MODELS.get(model_name, None)
        assert model_path is not None, f"Model {self._model_name} not found."

        self._model, self._images_transforms = longclip.load(model_path)
        self._tokenizer = longclip.tokenize
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

        texts_tokenized = torch.cat([self._tokenizer(text) for text in texts])
        texts_tokenized = texts_tokenized.to(self.device)
        texts_embeddings = self._model.encode_text(texts_tokenized)
        texts_embeddings = self.norm_if_needed(texts_embeddings, normalize)

        images_tensor = torch.stack(
            [self._images_transforms(img) for img in images_pil]
        ).to(self.device)
        images_embeddings = self._model.encode_image(images_tensor)
        images_embeddings = self.norm_if_needed(images_embeddings, normalize)

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

        images_tensor = torch.stack(
            [self._images_transforms(img) for img in images_pil]
        ).to(self.device)
        images_embeddings = self._model.encode_image(images_tensor)
        images_embeddings = self.norm_if_needed(images_embeddings, normalize)

        return {"image_embeddings": images_embeddings}
