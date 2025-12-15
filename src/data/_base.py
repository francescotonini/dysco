from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        name: str,
        root_dir: str,
        split: Optional[str] = None,
        transforms: Optional[torchvision.transforms.Compose] = None,
        images_transforms: Optional[torchvision.transforms.Compose] = None,
        targets_transforms: Optional[torchvision.transforms.Compose] = None,
    ) -> None:
        self._name = name
        self._root_dir = root_dir
        self._split = split
        self._transforms = transforms
        self._images_transforms = images_transforms
        self._targets_transforms = targets_transforms

        self._objects = []
        self._verbs = []
        self._int_obj_verb_id_matrix = []
        self._num_annotations_per_interaction = []
        self._rare_int_ids = []
        self._non_rare_int_ids = []
        self._objects_verbs_to_interaction_id = None
        self._num_annotations_per_object = None
        self._num_annotations_per_verb = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def split(self) -> Optional[str]:
        return self._split

    @property
    def transforms(self) -> torchvision.transforms.Compose:
        return self._transforms

    @transforms.setter
    def transforms(self, value: torchvision.transforms.Compose):
        self._transforms = value

    @property
    def images_transforms(self) -> torchvision.transforms.Compose:
        return self._images_transforms

    @images_transforms.setter
    def images_transforms(self, value: torchvision.transforms.Compose):
        self._images_transforms = value

    @property
    def targets_transforms(self) -> torchvision.transforms.Compose:
        return self._targets_transforms

    @targets_transforms.setter
    def targets_transforms(self, value: torchvision.transforms.Compose):
        self._targets_transforms = value

    @property
    def person_idx(self) -> int:
        return self.objects_name.index("person")

    @property
    def objects(self) -> list[Tuple[int, str]]:
        return list(zip(self.objects_id, self.objects_name))

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    @property
    def objects_name(self) -> list[str]:
        return self._objects

    @property
    def objects_id(self) -> list[int]:
        return list(range(len(self.objects_name)))

    @property
    def verbs(self) -> list[Tuple[int, str]]:
        return list(zip(self.verbs_id, self.verbs_name))

    @property
    def num_verbs(self) -> int:
        return len(self.verbs_name)

    @property
    def verbs_name(self) -> list[str]:
        return self._verbs

    @property
    def verbs_id(self) -> list[int]:
        return list(range(len(self.verbs_name)))

    @property
    def int_obj_verbs_id_matrix(self) -> List[Tuple[int, int, int]]:
        return self._int_obj_verb_id_matrix

    @property
    def objects_verbs_to_interaction_id(self) -> List[List[int]]:
        if self._objects_verbs_to_interaction_id is None:
            self._objects_verbs_to_interaction_id = torch.full(
                (self.num_objects, self.num_verbs), -1
            )
            for int_id, obj_id, verb_id in self._int_obj_verb_id_matrix:
                self._objects_verbs_to_interaction_id[obj_id, verb_id] = int_id

        return self._objects_verbs_to_interaction_id

    @property
    def num_annotations_per_object(self) -> List[int]:
        if self._num_annotations_per_object is None:
            self._num_annotations_per_object = [0 for _ in range(self.num_objects)]

            for int_id, obj_id, _ in self._int_obj_verb_id_matrix:
                self._num_annotations_per_object[obj_id] += self._num_annotations_per_interaction[
                    int_id
                ]

        return self._num_annotations_per_object

    @property
    def num_annotations_per_verb(self) -> List[int]:
        if self._num_annotations_per_verb is None:
            self._num_annotations_per_verb = [0 for _ in range(self.num_verbs)]

            for int_id, _, verb_id in self._int_obj_verb_id_matrix:
                self._num_annotations_per_verb[verb_id] += self._num_annotations_per_interaction[
                    int_id
                ]

        return self._num_annotations_per_verb

    def object_to_verbs(self, object_id) -> List[int]:
        return [
            verb_id for _, obj_id, verb_id in self._int_obj_verb_id_matrix if obj_id == object_id
        ]

    def verb_to_objects(self, verb_id) -> List[int]:
        return [
            obj_id for _, obj_id, _verb_id in self._int_obj_verb_id_matrix if _verb_id == verb_id
        ]

    @property
    def objects_to_interactions(self) -> List[List[int]]:
        return self._objects_to_interactions

    @property
    def objects_to_verbs(self) -> List[List[int]]:
        return self._objects_to_verbs

    @property
    def verbs_to_interactions(self) -> List[List[int]]:
        return self._verbs_to_interactions

    @property
    def interactions_id(self) -> list[Tuple[int, int]]:
        return [(obj_id, verb_id) for _, obj_id, verb_id in self.int_obj_verbs_id_matrix]

    @property
    def interactions_name(self) -> list[Tuple[str, str]]:
        return [
            (self.objects_name[obj_id], self.verbs_name[verb_id])
            for obj_id, verb_id in self.interactions_id
        ]

    @property
    def interactions_to_verbs(self) -> List[List[int]]:
        return torch.tensor([verb_id for _, verb_id in self.interactions_id])

    @property
    def interactions_to_objects(self) -> List[List[int]]:
        return torch.tensor([obj_id for obj_id, _ in self.interactions_id])

    @property
    def interactions(self) -> list[Tuple[Tuple[int, int], Tuple[str, str]]]:
        return list(zip(self.interactions_id, self.interactions_name))

    @property
    def num_interactions(self) -> int:
        return len(self.interactions)

    @property
    def num_annotations_per_interaction(self) -> List[int]:
        return self._num_annotations_per_interaction.copy()

    @property
    def rare_interactions_id(self) -> List[int]:
        return self._rare_int_ids

    @property
    def non_rare_interactions_id(self) -> List[int]:
        return self._non_rare_int_ids

    def setup(self, **kwargs) -> None:
        raise NotImplementedError("You must implement the setup method.")

    def __len__(self) -> int:
        raise NotImplementedError("You must implement the __len__ method.")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("You must implement the __getitem__ method.")
