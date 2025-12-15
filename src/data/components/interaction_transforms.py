import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import ImageDraw, ImageFilter

from src.utils.img_ops import bbox_overflow


class TargetReverseBlur:
    """Blur the image except the area given by the union bbox of human and object."""

    def __init__(self, shape="bbox", bbox_overflow_coefficient=0.0, blur_factor=100):
        assert shape in ["bbox"]  # TODO: "circle"
        self.shape = shape
        self.bbox_overflow_coefficient = bbox_overflow_coefficient
        self.blur_factor = blur_factor

    def __call__(self, image, target):
        human_bbox = target["human_bbox"]
        object_bbox = target["object_bbox"]
        union_bbox = torch.tensor(
            [
                torch.min(human_bbox[0], object_bbox[0]),
                torch.min(human_bbox[1], object_bbox[1]),
                torch.max(human_bbox[2], object_bbox[2]),
                torch.max(human_bbox[3], object_bbox[3]),
            ],
            device=human_bbox.device,
        )
        union_bbox = bbox_overflow(union_bbox, self.bbox_overflow_coefficient)
        union_bbox = [int(x) for x in union_bbox]

        # Blur the entire image except the union bbox
        if self.shape == "bbox":
            cropped_image = image.crop(union_bbox)

        image = image.filter(ImageFilter.GaussianBlur(self.blur_factor))
        image.paste(cropped_image, union_bbox)

        return image, target


class TargetBoundingBox:
    """Add a bounding box on the union bbox of human and object."""

    def __init__(
        self,
        color="red",
        width=3,
        shape="bbox",
        selection_type="union",
        bbox_overflow_coefficient=0.0,
    ):
        assert shape in ["bbox", "circle"]
        assert selection_type in ["union", "human", "object", "human_object"]

        self.color = color
        self.width = width
        self.shape = shape
        self.selection_type = selection_type
        self.bbox_overflow_coefficient = bbox_overflow_coefficient

    def __call__(self, image, target):
        human_bbox = target["human_bbox"]
        object_bbox = target["object_bbox"]

        draw_fn = self._draw_bbox if self.shape == "bbox" else self._draw_circle

        if "human" in self.selection_type:
            human_bbox = bbox_overflow(human_bbox, self.bbox_overflow_coefficient)

            draw_fn(image, human_bbox)
        if "object" in self.selection_type:
            object_bbox = bbox_overflow(object_bbox, self.bbox_overflow_coefficient)

            draw_fn(image, object_bbox)
        if self.selection_type == "union":
            union_bbox = torch.tensor(
                [
                    torch.min(human_bbox[0], object_bbox[0]),
                    torch.min(human_bbox[1], object_bbox[1]),
                    torch.max(human_bbox[2], object_bbox[2]),
                    torch.max(human_bbox[3], object_bbox[3]),
                ],
                device=human_bbox.device,
            )
            union_bbox = bbox_overflow(union_bbox, self.bbox_overflow_coefficient)

            draw_fn(image, union_bbox)

        return image, target

    def _draw_bbox(self, image, bbox):
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox.tolist(), outline=self.color, width=self.width)

    def _draw_circle(self, image, bbox):
        draw = ImageDraw.Draw(image)
        draw.ellipse(bbox.tolist(), outline=self.color, width=self.width)


class TargetCrop:
    """Crops the image given the union bbox of human and object."""

    def __init__(self, bbox_overflow_coefficient=0.0):
        self.bbox_overflow_coefficient = bbox_overflow_coefficient

    def __call__(self, image, target):
        human_bbox = target["human_bbox"]
        object_bbox = target["object_bbox"]
        union_bbox = torch.tensor(
            [
                torch.min(human_bbox[0], object_bbox[0]),
                torch.min(human_bbox[1], object_bbox[1]),
                torch.max(human_bbox[2], object_bbox[2]),
                torch.max(human_bbox[3], object_bbox[3]),
            ],
            device=human_bbox.device,
        )
        union_bbox = bbox_overflow(union_bbox, self.bbox_overflow_coefficient)
        target["union_bbox"] = union_bbox

        # Adjust bboxes according to the cropping
        for key in ["human_bbox", "object_bbox"]:
            if key in target:
                target[key][0] -= union_bbox[0]
                target[key][1] -= union_bbox[1]
                target[key][2] -= union_bbox[0]
                target[key][3] -= union_bbox[1]

        cropped_image = image.crop(union_bbox.tolist())
        target["image_size"] = torch.tensor(
            cropped_image.size,
            device=target["image_size"].device,
            dtype=target["image_size"].dtype,
        )

        return cropped_image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class SquarePad:
    def __call__(self, image, target):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)

        # Adjust bboxes according to the padding
        for key in ["human_bbox", "object_bbox", "union_bbox"]:
            if key in target:
                target[key][0] += hp
                target[key][1] += vp
                target[key][2] += hp
                target[key][3] += vp

        image_pad = F.pad(image, padding, 0, "constant")
        target["image_size"] = torch.tensor(
            image_pad.size, device=target["image_size"].device, dtype=target["image_size"].dtype
        )

        return image_pad, target


class NormalizeBoxes:
    def __call__(self, image, target):
        w, h = image.size

        for key in ["human_bbox", "object_bbox", "union_bbox"]:
            if key not in target:
                continue

            target[f"{key}_norm"] = target[key].clone()
            target[f"{key}_norm"][0] /= w
            target[f"{key}_norm"][1] /= h
            target[f"{key}_norm"][2] /= w
            target[f"{key}_norm"][3] /= h

        return image, target
