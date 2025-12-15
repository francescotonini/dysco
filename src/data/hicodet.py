import os

import torch
import torchvision
from PIL import Image

from src.data._base import BaseDataset
from src.utils import get_logger
from src.utils.hicodet_ops import (
    COCO_OBJECTS,
    HICODET_OBJECTS_IDXS_TO_COCO_IDXS,
)

log = get_logger(__name__)


class HICODET(BaseDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(name="HICODET", *args, **kwargs)

        assert self.split in ["test", "train"], f"Unknown HICODET split {self.split}"
        assert self.targets_transforms is None, "Targets transforms are not supported."

    @staticmethod
    def interaction_to_str(object_name, verb_name):
        return f"{object_name}_{verb_name}"

    def setup(self, **kwargs) -> None:
        if "limit_samples" in kwargs:
            limit_samples = kwargs["limit_samples"]
        else:
            limit_samples = None

        self._images_dir = os.path.join(
            self.root_dir,
            "images",
            {
                "test": "test2015",
                "train": "train2015",
            }[self.split],
        )
        self._annotations_path = os.path.join(
            self.root_dir, "annotations", f"instances_{self.split}2015.json"
        )

        # Process annotations
        with open(self._annotations_path) as f:
            self._raw_annotations = json.load(f)

        # Remap objects idxs to COCO if necessary
        if self._raw_annotations["objects"] != COCO_OBJECTS:
            log.warning("Annotations are not in COCO format! Remapping...")

            self._raw_annotations["objects"] = COCO_OBJECTS
            for correspondence in self._raw_annotations["correspondence"]:
                correspondence[1] = HICODET_OBJECTS_IDXS_TO_COCO_IDXS[correspondence[1]]

            for sample_idx, sample in enumerate(self._raw_annotations["annotation"]):
                sample["object"] = [
                    HICODET_OBJECTS_IDXS_TO_COCO_IDXS[obj_id] for obj_id in sample["object"]
                ]

        self._annotations = self._raw_annotations["annotation"]
        self._filenames = self._raw_annotations["filenames"]
        self._image_sizes = self._raw_annotations["size"]
        self._int_obj_verb_id_matrix = self._raw_annotations["correspondence"]
        self._objects = self._raw_annotations["objects"]
        self._verbs = self._raw_annotations["verbs"]
        self._rare_int_ids = self._raw_annotations["rare"]
        self._non_rare_int_ids = self._raw_annotations["non_rare"]

        idxs = list(range(len(self._filenames)))
        idxs_empty = self._raw_annotations["empty"]

        # Get image idxs and remove empty idxs
        for idx_empty in idxs_empty:
            idxs.remove(idx_empty)

        if limit_samples is not None and limit_samples > len(idxs):
            # If limit_samples is greater than the number of samples, set it to None
            log.warning(
                f"Limit samples {limit_samples} is greater than the number of samples ({len(idxs)}). Ignoring."
            )
            limit_samples = None

        num_annotations = [0 for _ in range(600)]
        for idx, annotation in enumerate(self._annotations):
            if limit_samples is not None and idx >= limit_samples:
                log.info(f"Limiting samples to {limit_samples}.")
                idxs = idxs[:limit_samples]

                break

            for hoi in annotation["hoi"]:
                num_annotations[hoi] += 1

        self._num_annotations_per_interaction = num_annotations
        self._idxs = idxs
        self._objects_to_interactions = [
            self.objects_verbs_to_interaction_id[obj_id][
                self.objects_verbs_to_interaction_id[obj_id] != -1
            ]
            for obj_id in range(self.num_objects)
        ]
        self._verbs_to_interactions = [
            self.objects_verbs_to_interaction_id[:, verb_id][
                self.objects_verbs_to_interaction_id[:, verb_id] != -1
            ]
            for verb_id in range(self.num_verbs)
        ]
        self._objects_to_verbs = [
            torch.tensor(
                [
                    self.interactions_id[interaction_id][1]
                    for interaction_id in self.objects_to_interactions[obj_id]
                ]
            )
            for obj_id in range(self.num_objects)
        ]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        idx = self._idxs[idx]
        annotation = self._annotations[idx]
        image_filename = self._filenames[idx]
        image_filepath = os.path.join(self._images_dir, image_filename)

        image_pil = Image.open(image_filepath).convert("RGB")
        image_size = torch.tensor(image_pil.size).long()

        verbs_id = []
        objects_id = []
        humans_bbox = []
        objects_bbox = []
        for human_bbox, object_bbox, object_id, verb_id in zip(
            annotation["boxes_h"],
            annotation["boxes_o"],
            annotation["object"],
            annotation["verb"],
        ):
            verbs_id.append(verb_id)
            objects_id.append(object_id)
            humans_bbox.append(human_bbox)
            objects_bbox.append(object_bbox)

        verbs_id = torch.tensor(verbs_id, dtype=torch.long)
        objects_id = torch.tensor(objects_id, dtype=torch.long)
        humans_bbox = torch.tensor(humans_bbox).long()
        humans_bbox[:, :2] -= 1
        objects_bbox = torch.tensor(objects_bbox).long()
        objects_bbox[:, :2] -= 1

        humans_bbox[:, 0] = torch.clamp(humans_bbox[:, 0], min=0, max=image_size[0])
        humans_bbox[:, 1] = torch.clamp(humans_bbox[:, 1], min=0, max=image_size[1])
        humans_bbox[:, 2] = torch.clamp(humans_bbox[:, 2], min=0, max=image_size[0])
        humans_bbox[:, 3] = torch.clamp(humans_bbox[:, 3], min=0, max=image_size[1])
        objects_bbox[:, 0] = torch.clamp(objects_bbox[:, 0], min=0, max=image_size[0])
        objects_bbox[:, 1] = torch.clamp(objects_bbox[:, 1], min=0, max=image_size[1])
        objects_bbox[:, 2] = torch.clamp(objects_bbox[:, 2], min=0, max=image_size[0])
        objects_bbox[:, 3] = torch.clamp(objects_bbox[:, 3], min=0, max=image_size[1])

        target = {
            "images_filename": image_filename,
            "images_filepath": image_filepath,
            "images_size": image_size,
            "verbs_id": verbs_id,
            "humans_bbox": humans_bbox,
            "objects_bbox": objects_bbox,
            "objects_id": objects_id,
        }

        if self.transforms:
            image_pil, target = self.transforms(image_pil, target)

        if self.images_transforms:
            image_tensor = self.images_transforms(image_pil)
        else:
            image_tensor = torchvision.transforms.functional.to_tensor(image_pil)

        target["images_tensor"] = image_tensor
        target["images_pil"] = image_pil

        return target
