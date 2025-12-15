import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import torch
import torchmetrics
import torchvision
import torchvision.transforms.functional
from lightning_utilities import apply_to_collection
from torchvision.ops import boxes as box_ops
import data.components.interaction_transforms as IT
from src.detectors.detr import DETR
from src.models.components.classifier import Classifier
from src.models.encoders._base import BaseEncoder
from src.trainer import Trainer
from src.utils import get_logger
from src.utils.hoi_map_metric import HOIMapMetric

log = get_logger(__name__)

def softmax_1(logits, dim=-1):
    max_logits, _ = torch.max(logits, dim=dim, keepdim=True)
    stabilized_logits = logits - max_logits
    exp_logits = torch.exp(stabilized_logits)
    sum_exp_logits = torch.sum(exp_logits, dim=dim, keepdim=True)
    softmax_output = exp_logits / (1 + sum_exp_logits)
    return softmax_output

class DYSCO(L.LightningModule):
    def __init__(
        self,
        *args,
        multimodal_encoder: BaseEncoder,
        interaction_transforms: Optional[IT.Compose] = None,
        detector: DETR = None,
        min_instances: int = 3,
        max_instances: int = 15,
        box_score_threshold: float = 0.2,
        nms_threshold: float = 0.5,
        vision_cache_path: Optional[str] = None,
        negative_vision_cache_path: Optional[str] = None,
        num_shots: int = 8,
        registry_cache_path: Optional[str] = None,
        rare_registry_cache_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._trainer: Optional[Trainer] = None
        self._metrics: Dict[str, torchmetrics.Metric] = {}

        self._multimodal_encoder: BaseEncoder = multimodal_encoder

        self._interaction_transforms: Optional[IT.Compose] = interaction_transforms
        self._detector: DETR = detector
        self._min_instances: int = min_instances
        self._max_instances: int = max_instances
        self._box_score_threshold: float = box_score_threshold
        self._nms_threshold: float = nms_threshold

        self._vision_cache_path: Path = Path(vision_cache_path)
        self._negative_vision_cache_path: Path = Path(negative_vision_cache_path)
        self._registry_cache_path: Path = Path(registry_cache_path)
        self._rare_registry_cache_path: Path = Path(rare_registry_cache_path)
        self._num_shots: int = num_shots

        self._classifier: Classifier = Classifier()

        self._registry_z = None
        self._rare_registry_z = None

    @property
    def interaction_transforms(self) -> Optional[IT.Compose]:
        return self._interaction_transforms

    @property
    def detector(self) -> DETR:
        return self._detector

    @property
    def num_parameters(self) -> Optional[Dict[str, int]]:
        return {"detector": sum(p.numel() for p in self._detector.parameters())}

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: Trainer):
        self._trainer = trainer

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Dict[str, torchmetrics.Metric]):
        self._metrics = metrics

    @property
    def device(self) -> torch.device:
        return self._multimodal_encoder.device

    @property
    def logit_scale(self) -> torch.Tensor:
        if hasattr(self._multimodal_encoder, "logit_scale"):
            return self._multimodal_encoder.logit_scale
        return torch.tensor(1.0, device=self.device)

    def encode_images(self, images: Any, normalize=True) -> torch.Tensor:
        return self._multimodal_encoder.get_vision_embeddings(images, normalize=normalize)[
            "image_embeddings"
        ]

    def get_bboxes(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        batch = copy.deepcopy(batch)
        batch["images_tensor"] = self.fabric.strategy.precision.convert_input(
            batch["images_tensor"]
        )
        return self.detector(batch)

    def get_interactions_pairings(self, preds_detector: Dict[str, Any]) -> Any:
        bs = len(preds_detector["boxes"])
        device = preds_detector["boxes"][0].device
        boxes = preds_detector["boxes"]
        labels = preds_detector["labels"]
        scores = preds_detector["scores"]

        pairings = []
        for b_idx in range(bs):
            keep = box_ops.batched_nms(
                boxes[b_idx], scores[b_idx], labels[b_idx], self._nms_threshold
            )
            boxes[b_idx] = boxes[b_idx][keep]
            scores[b_idx] = scores[b_idx][keep]
            labels[b_idx] = labels[b_idx][keep]
            keep = torch.nonzero(scores[b_idx] >= self._box_score_threshold).squeeze(1)

            human_mask = labels[b_idx] == self.trainer.dataset.person_idx
            human_idx = torch.nonzero(human_mask).squeeze(1)
            object_idx = torch.nonzero(human_mask == 0).squeeze(1)
            num_humans = human_mask[keep].sum()
            num_objects = len(keep) - num_humans

            if num_humans < self._min_instances:
                keep_humans = scores[b_idx][human_idx].argsort(descending=True)[
                    : self._min_instances
                ]
                keep_humans = human_idx[keep_humans]
            elif num_humans > self._max_instances:
                keep_humans = scores[b_idx][human_idx].argsort(descending=True)[
                    : self._max_instances
                ]
                keep_humans = human_idx[keep_humans]
            else:
                keep_humans = torch.nonzero(human_mask[keep]).squeeze(1)
                keep_humans = keep[keep_humans]

            if num_objects < self._min_instances:
                keep_objects = scores[b_idx][object_idx].argsort(descending=True)[
                    : self._min_instances
                ]
                keep_objects = object_idx[keep_objects]
            elif num_objects > self._max_instances:
                keep_objects = scores[b_idx][object_idx].argsort(descending=True)[
                    : self._max_instances
                ]
                keep_objects = object_idx[keep_objects]
            else:
                keep_objects = torch.nonzero(human_mask[keep] == 0).squeeze(1)
                keep_objects = keep[keep_objects]

            keep = torch.cat([keep_humans, keep_objects])
            boxes[b_idx] = boxes[b_idx][keep]
            scores[b_idx] = scores[b_idx][keep]
            labels[b_idx] = labels[b_idx][keep]

            is_human = labels[b_idx] == self.trainer.dataset.person_idx
            num_humans = torch.sum(is_human)
            num_boxes = len(boxes[b_idx])

            if not torch.all(labels[b_idx][:num_humans] == self.trainer.dataset.person_idx):
                humans_idx = torch.nonzero(is_human).squeeze(1)
                objects_idx = torch.nonzero(is_human == 0).squeeze(1)
                keep = torch.cat([humans_idx, objects_idx])
                boxes[b_idx] = boxes[b_idx][keep]
                scores[b_idx] = scores[b_idx][keep]
                labels[b_idx] = labels[b_idx][keep]

            if num_humans == 0 or num_boxes <= 1:
                pairings.append(torch.empty(0, 2, device=device, dtype=torch.long))
                continue

            humans_idx, objects_idx = torch.meshgrid(
                torch.arange(num_boxes, device=device), torch.arange(num_boxes, device=device)
            )
            humans_idx, objects_idx = torch.nonzero(
                torch.logical_and(humans_idx != objects_idx, humans_idx < num_humans)
            ).unbind(1)

            if len(humans_idx) == 0:
                raise ValueError("There are no valid human-object pairs")

            pairings.append(torch.stack([humans_idx, objects_idx], dim=1))

        return pairings

    def configure_metrics(self):
        return {
            "test_hoi_non_rare": HOIMapMetric(
                interactions_name=self.trainer.dataset.interactions_name,
                num_interactions=self.trainer.dataset.num_interactions,
                objects_verbs_to_interaction_id=self.trainer.dataset.objects_verbs_to_interaction_id,
                num_annotations_per_interaction=self.trainer.dataset.num_annotations_per_interaction,
                interactions_id=self.trainer.dataset.non_rare_interactions_id,
            ),
            "test_hoi_rare": HOIMapMetric(
                interactions_name=self.trainer.dataset.interactions_name,
                num_interactions=self.trainer.dataset.num_interactions,
                objects_verbs_to_interaction_id=self.trainer.dataset.objects_verbs_to_interaction_id,
                num_annotations_per_interaction=self.trainer.dataset.num_annotations_per_interaction,
                interactions_id=self.trainer.dataset.rare_interactions_id,
            ),
            "test_hoi": HOIMapMetric(
                interactions_name=self.trainer.dataset.interactions_name,
                num_interactions=self.trainer.dataset.num_interactions,
                objects_verbs_to_interaction_id=self.trainer.dataset.objects_verbs_to_interaction_id,
                num_annotations_per_interaction=self.trainer.dataset.num_annotations_per_interaction,
            ),
        }

    def setup(self, stage: str) -> None:
        self.metrics = apply_to_collection(
            self.configure_metrics(), torchmetrics.Metric, lambda x: self.fabric.to_device(x)
        )
        return super().setup(stage)

    def on_test_epoch_start(self) -> None:
        (
            self._human_object_vision_cache,
            self._interactions_vision_cache,
            self._human_object_labels_cache,
            self._human_object_lens,
        ) = self.load_cache(self._vision_cache_path)
        self._interactions_labels_cache = self._human_object_labels_cache
        self._interactions_lens = self._human_object_lens

        log.info("Loading negative vision cache")
        (
            self._human_object_negative_vision_cache,
            self._interactions_negative_vision_cache,
            self._human_object_negative_labels_cache,
            self._human_object_negative_lens,
        ) = self.load_cache(
            self._negative_vision_cache_path,
            negative_as_opposite_of_positive=True,
        )
        self._interactions_negative_labels_cache = self._human_object_negative_labels_cache
        self._interactions_negative_lens = self._human_object_negative_lens

        log.info(f"Loading precomputed registry from {self._registry_cache_path}")
        self._registry_z = torch.load(self._registry_cache_path, map_location=self.device)

        log.info(f"Loading precomputed rare registry from {self._rare_registry_cache_path}")
        self._rare_registry_z = torch.load(self._rare_registry_cache_path, map_location=self.device)

    def load_cache(
        self,
        cache_path: Path,
        negative_as_opposite_of_positive: bool = False,
    ) -> Any:
        log.info(f"Loading cache from {cache_path}")

        raw_features = torch.load(cache_path, map_location="cpu", weights_only=False)

        num_cache_entries = self.trainer.dataset.num_interactions

        human_vision_embeddings = [[] for _ in range(num_cache_entries)]
        object_vision_embeddings = [[] for _ in range(num_cache_entries)]
        interaction_vision_embeddings = [[] for _ in range(num_cache_entries)]
        real_verbs = [[] for _ in range(num_cache_entries)]

        cache_human_object_vision_features = []
        cache_interaction_vision_features = []
        cache_labels = []

        for filename, features in raw_features["features"].items():
            interaction_pairings = features["interactions_pairings"]
            objects_id = features["pred_labels"][interaction_pairings[:, 1]]
            verbs_id = features["verbs_id"]
            interactions_id = self.trainer.dataset.objects_verbs_to_interaction_id[
                objects_id.cpu(), verbs_id.cpu()
            ]

            num_ho_pair = len(interaction_pairings)
            features["real_verbs"] = torch.zeros(num_ho_pair, num_cache_entries)
            for i in range(num_ho_pair):
                features["real_verbs"][i][interactions_id[i]] = 1

            pred_human_boxes = features["pred_boxes"][interaction_pairings[:, 0]]
            pred_object_boxes = features["pred_boxes"][interaction_pairings[:, 1]]
            pred_objects_id = features["pred_labels"][interaction_pairings[:, 1]]
            boxes_h_iou = torchvision.ops.box_iou(pred_human_boxes, pred_human_boxes)
            boxes_o_iou = torchvision.ops.box_iou(pred_object_boxes, pred_object_boxes)
            for i in range(num_ho_pair):
                idx_h = boxes_h_iou[i] > 0.5
                idx_o = torch.logical_and(
                    boxes_o_iou[i] > 0.5,
                    pred_objects_id == pred_objects_id[i],
                )
                idx_ho = torch.logical_and(idx_h, idx_o)

                features["real_verbs"][i] = torch.max(
                    features["real_verbs"][idx_ho], dim=0
                ).values

                if negative_as_opposite_of_positive:
                    valid_interaction_ids = (
                        self.trainer.dataset.objects_to_interactions[pred_objects_id[i]]
                    )
                    features_mask = features["real_verbs"][i][valid_interaction_ids] != 0
                    features["real_verbs"][i][valid_interaction_ids][features_mask] = 1
                    features["real_verbs"][i][valid_interaction_ids] = (
                        1 - features["real_verbs"][i][valid_interaction_ids]
                    )

            pred_human_vision_features = features["pred_boxes_image_features"][
                interaction_pairings[:, 0]
            ]
            pred_object_vision_features = features["pred_boxes_image_features"][
                interaction_pairings[:, 1]
            ]
            pred_interaction_vision_features = features["pred_interactions_image_features"]

            for idx, interaction_id in enumerate(interactions_id):
                entry_id = interaction_id

                interaction_vision_embeddings[entry_id].append(
                    torch.nn.functional.normalize(
                        pred_interaction_vision_features[idx], dim=-1
                    ).squeeze(0)
                )
                human_vision_embeddings[entry_id].append(
                    torch.nn.functional.normalize(
                        pred_human_vision_features[idx], dim=-1
                    ).squeeze(0)
                )
                object_vision_embeddings[entry_id].append(
                    torch.nn.functional.normalize(
                        pred_object_vision_features[idx], dim=-1
                    ).squeeze(0)
                )
                real_verbs[entry_id].append(features["real_verbs"][idx])

        human_vision_embeddings = [
            torch.stack(h) if len(h) > 0 else [] for h in human_vision_embeddings
        ]
        object_vision_embeddings = [
            torch.stack(o) if len(o) > 0 else [] for o in object_vision_embeddings
        ]
        interaction_vision_embeddings = [
            torch.stack(i) if len(i) > 0 else [] for i in interaction_vision_embeddings
        ]

        for entry_id, _ in enumerate(self.trainer.dataset.interactions_id):
            if (
                len(human_vision_embeddings[entry_id]) == 0
                or len(object_vision_embeddings[entry_id]) == 0
            ):
                log.warning(f"Skipping {entry_id} due to empty vision embeddings")
                continue

            num_samples = human_vision_embeddings[entry_id].shape[0]
            if self._num_shots > 0:
                num_to_select = min(self._num_shots, num_samples)
            else:
                num_to_select = num_samples
            idxs = torch.randperm(num_samples)[:num_to_select]

            human_object_vision_embeddings = torch.cat(
                [human_vision_embeddings[entry_id], object_vision_embeddings[entry_id]],
                dim=-1,
            )

            cache_human_object_vision_features.append(human_object_vision_embeddings[idxs])
            cache_interaction_vision_features.append(
                interaction_vision_embeddings[entry_id][idxs]
            )
            cache_labels.append(torch.stack(real_verbs[entry_id])[idxs])

        human_object_vision_cache = self.fabric.to_device(
            torch.cat(cache_human_object_vision_features).float()
        )
        interaction_vision_cache = self.fabric.to_device(
            torch.cat(cache_interaction_vision_features).float()
        )
        labels_cache = self.fabric.to_device(torch.cat(cache_labels).float())
        interactions_len = self.fabric.to_device(torch.sum(labels_cache, dim=0).float())

        return (
            human_object_vision_cache,
            interaction_vision_cache,
            labels_cache,
            interactions_len,
        )

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        bs = len(batch["images_tensor"])

        preds_detector = self.get_bboxes(batch, batch_idx)
        batch_interactions_pairings = self.get_interactions_pairings(preds_detector)

        all_batch_crops = []
        outputs_and_splits = []
        for sample_idx in range(bs):
            output, crops = self.get_features(
                batch, sample_idx, batch_interactions_pairings[sample_idx], preds_detector,
                _precomputed_features=torch.empty(0),  # skip encoding
            )
            n = len(crops) if crops else 0
            all_batch_crops.extend(crops if crops else [])
            outputs_and_splits.append((output, n))

        if len(all_batch_crops) > 0:
            max_chunk = 64
            if len(all_batch_crops) <= max_chunk:
                all_batch_features = self.encode_images(all_batch_crops)
            else:
                chunks = [
                    all_batch_crops[i : i + max_chunk]
                    for i in range(0, len(all_batch_crops), max_chunk)
                ]
                all_batch_features = torch.cat(
                    [self.encode_images(chunk) for chunk in chunks], dim=0
                )
        else:
            all_batch_features = None

        preds = {}
        offset = 0
        for sample_idx in range(bs):
            image_filename = batch["images_filename"][sample_idx]
            output, n_crops = outputs_and_splits[sample_idx]

            if all_batch_features is not None and n_crops > 0:
                features = all_batch_features[offset : offset + n_crops]
                self._fill_output_features(output, features)
            offset += n_crops

            pred_boxes = output["pred_boxes"]
            pred_boxes_id = output["pred_boxes_id"]
            pred_boxes_scores = output["pred_boxes_scores"]
            interactions_pairings = output["interactions_pairings"]
            pred_interactions_features = output["pred_interactions_features"]

            pred_interactions_scores = []
            for _sample_idx in range(len(pred_interactions_features)):
                pred_interactions_scores.append(
                    self.get_interactions_scores(output, batch, _sample_idx, sample_idx)
                )
            pred_interactions_scores = torch.stack(pred_interactions_scores)

            pred_prior_scores = self.get_prior_scores(
                interactions_pairings[:, 0],
                interactions_pairings[:, 1],
                pred_boxes_scores.float(),
                pred_boxes_id,
            )
            pred_prior_scores = pred_prior_scores.prod(dim=0)
            pred_idx, pred_interactions_idx = torch.nonzero(pred_prior_scores).unbind(1)

            pred_scores = (
                pred_interactions_scores[pred_idx, pred_interactions_idx]
                * pred_prior_scores[pred_idx, pred_interactions_idx]
            )
            pred_interactions_pairings = torch.cat(
                [
                    interactions_pairings[:, 0][pred_idx].unsqueeze(-1),
                    interactions_pairings[:, 1][pred_idx].unsqueeze(-1),
                ],
                dim=1,
            )

            preds[image_filename] = {
                "boxes": pred_boxes,
                "boxes_id": pred_boxes_id,
                "boxes_scores": pred_boxes_scores,
                "interactions_pairings": pred_interactions_pairings,
                "interactions_scores": pred_scores,
                "interactions_verbs_id": torch.tensor(
                    [
                        self.trainer.dataset.interactions_id[idx][1]
                        for idx in pred_interactions_idx
                    ],
                    device=self.device,
                    dtype=torch.long,
                ),
            }

        return preds

    def get_features(
        self,
        batch: Dict[str, Any],
        sample_idx: int,
        interactions_pairings: Dict[str, Any],
        preds_detector: Dict[str, Any],
        _precomputed_features: Optional[torch.Tensor] = None,
    ):
        pred_boxes = preds_detector["boxes"][sample_idx]
        pred_boxes_id = preds_detector["labels"][sample_idx]
        pred_boxes_scores = preds_detector["scores"][sample_idx]
        pred_humans_bbox = pred_boxes[interactions_pairings[:, 0]]
        pred_humans_score = pred_boxes_scores[interactions_pairings[:, 0]]
        pred_objects_bbox = pred_boxes[interactions_pairings[:, 1]]
        pred_objects_score = pred_boxes_scores[interactions_pairings[:, 1]]
        pred_objects_id = pred_boxes_id[interactions_pairings[:, 1]]
        image_size = batch["images_size"][sample_idx]
        image_pil = batch["images_pil"][sample_idx]

        boxes_crop_pil = [image_pil.crop(pred_box.tolist()) for pred_box in pred_boxes]
        n_boxes = len(boxes_crop_pil)

        images_crop_pil = []
        for human_bbox, human_score, object_bbox, object_score, object_id in zip(
            pred_humans_bbox,
            pred_humans_score,
            pred_objects_bbox,
            pred_objects_score,
            pred_objects_id,
        ):
            target = {
                "human_bbox": human_bbox.clone(),
                "human_score": human_score.clone(),
                "object_bbox": object_bbox.clone(),
                "object_score": object_score.clone(),
                "object_id": object_id.clone(),
                "image_size": image_size.clone(),
            }

            if self.interaction_transforms is not None:
                image_crop_pil, new_target = self.interaction_transforms(image_pil, target)
            else:
                image_crop_pil = image_pil
                new_target = target

            images_crop_pil.append(image_crop_pil)

        n_interactions = len(images_crop_pil)
        all_crops = boxes_crop_pil + images_crop_pil

        if _precomputed_features is not None and len(_precomputed_features) > 0:
            all_features = _precomputed_features
        elif len(all_crops) > 0 and _precomputed_features is None:
            all_features = self.encode_images(all_crops)
        else:
            all_features = None

        if all_features is not None:
            pred_boxes_features = all_features[:n_boxes]
            pred_interactions_features = all_features[n_boxes : n_boxes + n_interactions]
        else:
            pred_boxes_features = torch.empty(0, device=self.device)
            pred_interactions_features = torch.empty(0, device=self.device)

        output = {
            "pred_boxes": pred_boxes,
            "pred_boxes_id": pred_boxes_id,
            "pred_boxes_scores": pred_boxes_scores,
            "pred_boxes_features": pred_boxes_features,
            "pred_interactions_features": pred_interactions_features,
            "interactions_pairings": interactions_pairings,
            "_crop_splits": (n_boxes, n_interactions),
        }

        return output, all_crops

    def _fill_output_features(self, output: Dict[str, Any], features: torch.Tensor):
        n_boxes, n_interactions = output["_crop_splits"]
        output["pred_boxes_features"] = features[:n_boxes]
        output["pred_interactions_features"] = features[n_boxes : n_boxes + n_interactions]

    def get_interactions_scores(
        self, output: Dict[str, Any], batch: Dict[str, Any], sample_idx: int, batch_idx: int
    ) -> torch.Tensor:
        interactions_pairings = output["interactions_pairings"][sample_idx : sample_idx + 1]

        pred_boxes_features = output["pred_boxes_features"]
        pred_interactions_features = output["pred_interactions_features"][
            sample_idx : sample_idx + 1
        ].float()
        pred_human_vision_features = (
            torch.stack([pred_boxes_features[i] for i, _ in interactions_pairings])
        ).float()
        pred_object_vision_features = (
            torch.stack([pred_boxes_features[j] for _, j in interactions_pairings])
        ).float()
        pred_human_object_vision_features = torch.cat(
            [pred_human_vision_features, pred_object_vision_features],
            dim=-1,
        ).float()

        object_id = output["pred_boxes_id"][output["interactions_pairings"][:, 1]][sample_idx]
        interactions_idx = (
            self.trainer.dataset.objects_to_interactions[object_id].sort().values
        ).to(self.device)
        interactions_idx_mask = torch.zeros(
            self.trainer.dataset.num_interactions, device=self.device
        ).bool()
        if interactions_idx.numel() > 0:
            interactions_idx_mask[interactions_idx] = True

        aff = pred_human_object_vision_features @ self._human_object_vision_cache.t()
        logits_human_object_vision = (
            aff @ self._human_object_labels_cache
            / self._human_object_labels_cache.sum(dim=0)
        ) / 2
        logits_human_object_vision *= 0.5
        logits_ho_neg = (
            (pred_human_object_vision_features @ self._human_object_negative_vision_cache.t())
            @ self._human_object_negative_labels_cache
            / self._human_object_negative_labels_cache.sum(dim=0)
        ).nan_to_num(0.0)
        logits_human_object_vision -= logits_ho_neg * 0.5

        aff = pred_interactions_features @ self._interactions_vision_cache.t()
        logits_interaction_vision = (
            aff @ self._interactions_labels_cache
            / self._interactions_labels_cache.sum(dim=0)
        )
        logits_interaction_vision *= 0.5
        logits_int_neg = (
            (pred_interactions_features @ self._interactions_negative_vision_cache.t())
            @ self._interactions_negative_labels_cache
            / self._interactions_negative_labels_cache.sum(dim=0)
        ).nan_to_num(0.0)
        logits_interaction_vision -= logits_int_neg * 0.5

        logits_registry = self._classifier(
            pred_interactions_features, self._registry_z, op="non_rare",
        ) * 1.0
        logits_rare_registry = self._classifier(
            pred_interactions_features, self._rare_registry_z, op="rare",
        ) * 1.0

        logits_concat = torch.cat(
            [
                logits_human_object_vision.clone(),
                logits_interaction_vision.clone(),
                logits_registry.clone(),
                logits_rare_registry.clone(),
            ],
            dim=0,
        )
        logits_concat /= 0.1
        logits_concat = logits_concat.nan_to_num_(float("-inf"))
        probs_heads = softmax_1(logits_concat, dim=0)

        logits = (
            logits_human_object_vision * (1 + probs_heads[0])
            + logits_interaction_vision * (1 + probs_heads[1])
            + logits_registry * (1 + probs_heads[2])
            + logits_rare_registry * (1 + probs_heads[3])
        ) / (0.5 + 0.5 + 1.0 + 1.0)

        logits[:, ~interactions_idx_mask] = float("-inf")

        probs = (logits / 0.05).softmax(dim=-1)
        return probs.squeeze(0)

    def get_prior_scores(
        self,
        pred_humans_idx: torch.Tensor,
        pred_objects_idx: torch.Tensor,
        scores: torch.Tensor,
        object_class: torch.Tensor,
    ) -> torch.Tensor:
        prior_humans_interaction_score = torch.zeros(
            len(pred_humans_idx),
            len(self.trainer.dataset.interactions),
            device=scores.device,
        ).float()
        prior_objects_interaction_score = torch.zeros_like(prior_humans_interaction_score).float()

        scores_human = scores[pred_humans_idx].pow(2.8)
        scores_object = scores[pred_objects_idx].pow(2.8)

        target_cls_idx = [
            self.trainer.dataset.objects_to_interactions[obj.item()]
            for obj in object_class[pred_objects_idx]
        ]

        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_humans_interaction_score[pair_idx, flat_target_idx] = scores_human[pair_idx].float()
        prior_objects_interaction_score[pair_idx, flat_target_idx] = scores_object[
            pair_idx
        ].float()

        return torch.stack([prior_humans_interaction_score, prior_objects_interaction_score])

    def on_test_batch_end(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        preds = {}
        for image_idx, image_filename in enumerate(batch["images_filename"]):
            if image_filename not in outputs:
                continue

            output = outputs[image_filename]
            boxes = output["boxes"]
            boxes_id = output["boxes_id"]
            boxes_scores = output["boxes_scores"]
            pairings = output["interactions_pairings"]
            interactions_verbs_id = output["interactions_verbs_id"]
            interactions_scores = output["interactions_scores"]

            if boxes.numel() == 0:
                log.warning(f"No boxes detected for {image_filename}")
                continue

            preds[image_filename] = {
                "boxes": boxes,
                "boxes_id": boxes_id,
                "boxes_scores": boxes_scores,
                "humans_bbox": boxes[pairings[:, 0]],
                "humans_score": boxes_scores[pairings[:, 0]],
                "humans_id": boxes_id[pairings[:, 0]],
                "objects_bbox": boxes[pairings[:, 1]],
                "objects_score": boxes_scores[pairings[:, 1]],
                "objects_id": boxes_id[pairings[:, 1]],
                "verbs_id": interactions_verbs_id,
                "scores": interactions_scores,
            }

        tgts = {}
        for image_idx, image_filename in enumerate(batch["images_filename"]):
            tgts[image_filename] = {
                "humans_bbox": batch["humans_bbox"][image_idx],
                "objects_bbox": batch["objects_bbox"][image_idx],
                "verbs_id": batch["verbs_id"][image_idx],
                "objects_id": batch["objects_id"][image_idx],
            }

        for k, metric in self.metrics.items():
            if "test_hoi" in k:
                metric.update(preds, tgts)
