import torch

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from packaging import version
from PIL import Image

if version.parse(torchvision.__version__) < version.parse("0.7"):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size

import torchvision.transforms.functional as F


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """Equivalent to nn.functional.interpolate, but with support for empty batch sizes.

    This will eventually be supported natively by PyTorch, and this class can go away.
    """
    if version.parse(torchvision.__version__) < version.parse("0.7"):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    # fields = ["labels", "area"]
    fields = ["labels", "object"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    # Crop human and object boxes
    if "boxes_h" in target:
        boxes = target["boxes_h"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes_h"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes_h")
    if "boxes_o" in target:
        boxes = target["boxes_o"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes_o"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes_o")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target or "boxes_h" in target or "boxes_o" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        elif "masks" in target:
            keep = target["masks"].flatten(1).any(1)
        else:
            cropped_bh = target["boxes_h"].reshape(-1, 2, 2)
            cropped_bo = target["boxes_o"].reshape(-1, 2, 2)
            keep = torch.logical_and(
                torch.all(cropped_bh[:, 1, :] > cropped_bo[:, 0, :], dim=1),
                torch.all(cropped_bo[:, 1, :] > cropped_bo[:, 0, :], dim=1),
            )

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor(
            [w, 0, w, 0]
        )
        target["boxes"] = boxes

    # Flip human and object boxes
    if "boxes_h" in target:
        boxes = target["boxes_h"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor(
            [w, 0, w, 0]
        )
        target["boxes_h"] = boxes
    if "boxes_o" in target:
        boxes = target["boxes_o"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor(
            [w, 0, w, 0]
        )
        target["boxes_o"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    # Resize human and object boxes
    if "boxes_h" in target:
        boxes = target["boxes_h"]
        scaled_boxes = boxes * torch.tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes_h"] = scaled_boxes
    if "boxes_o" in target:
        boxes = target["boxes_o"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes_o"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0] > 0.5
        )

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(target["masks"], (0, padding[0], 0, padding[1]))
    return padded_image, target


def bbox_overflow(bbox, overflow_coeff):
    if isinstance(bbox, torch.Tensor):
        overflow_coeff = torch.tensor(overflow_coeff, device=bbox.device)

    # So we can go back to the original dtype
    bbox_dtype = bbox.dtype

    bbox = bbox.float()

    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    bbox[0] -= bbox_width * overflow_coeff
    bbox[1] -= bbox_height * overflow_coeff
    bbox[2] += bbox_width * overflow_coeff
    bbox[3] += bbox_height * overflow_coeff

    bbox = bbox.to(bbox_dtype)

    return bbox


def expand2square(pil_img):
    width, height = pil_img.size

    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), (0, 0, 0))
        result.paste(pil_img, (0, (width - height) // 2))

        return result
    else:
        result = Image.new(pil_img.mode, (height, height), (0, 0, 0))
        result.paste(pil_img, ((height - width) // 2, 0))

        return result
