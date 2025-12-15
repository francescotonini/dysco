# Credits: fredzzhang

from typing import Any, Dict, List, Optional, Tuple

import torch
from torchmetrics import Metric
from torchvision.ops import box_iou

from src.utils import get_logger

log = get_logger(__name__)


class HOIMapMetric(Metric):
    def __init__(
        self,
        *args,
        num_interactions: int,
        objects_verbs_to_interaction_id: Dict[Tuple[int, int], int],
        num_annotations_per_interaction: List[int],
        interactions_name: List[Tuple[str, str]],
        iou_threshold: Optional[float] = 0.5,
        interactions_id: List[int] = None,
        ap_method: str = "11P",
        precision: int = 64,
        nproc: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert ap_method in ["AUC", "11P", "INT"], "Invalid ap_method"
        assert (
            len(num_annotations_per_interaction) == num_interactions
        ), "Invalid number of interactions"

        self.num_interactions = num_interactions
        self.num_annotations_per_interaction = num_annotations_per_interaction
        self.interactions_name = interactions_name
        self.iou_threshold = iou_threshold
        self.interactions_id = (
            interactions_id if interactions_id is not None else list(range(num_interactions))
        )
        self.ap_method = ap_method
        if not hasattr(self, "dtype"):
            self.dtype = eval(f"torch.float{precision}")
        self.nproc = nproc

        self.add_state(
            "objects_verbs_to_interaction_id",
            default=objects_verbs_to_interaction_id,
            dist_reduce_fx=None,
        )

        self.aps = None
        self.max_recs = None

        self.add_state("all_pred_scores", default=[], dist_reduce_fx="cat")
        self.add_state("all_pred_interactions_id", default=[], dist_reduce_fx="cat")
        self.add_state("all_pred_tgt_matchings", default=[], dist_reduce_fx="cat")

        # NOTE: This snippet is from the original implementation of the metric, which did not support distributed evaluation.
        # Now, these attributes are used to store the output and labels only during the `compute` step,
        # when all predictions from each GPU are gathered.
        self.output = []
        self.labels = []
        for _ in range(num_interactions):
            self.output.append([])
            self.labels.append([])

    def update(self, preds: dict[Any], tgts: dict[Any]):
        # Keep track of the scores, predictions and labels for each item in the batch
        all_pred_scores = []
        all_pred_interactions_id = []
        all_pred_tgt_matchings = []

        # Get target keys
        tgt_keys = tgts.keys()
        for tgt_key in tgt_keys:
            tgt = tgts.get(tgt_key, None)
            pred = preds.get(tgt_key, None)

            # If preds is empty, skip the iteration
            if not pred:
                log.debug(f"Empty predictions for {tgt_key}")

                continue

            pred_human_bboxes = pred["humans_bbox"]
            pred_object_bboxes = pred["objects_bbox"]
            pred_scores = pred["scores"]
            pred_verb_ids = pred["verbs_id"]
            pred_object_ids = pred["objects_id"]

            # Get interactions id
            pred_interactions_id = self.objects_verbs_to_interaction_id[
                (pred_object_ids, pred_verb_ids)
            ]

            tgt_human_bboxes = tgt["humans_bbox"]
            tgt_object_bboxes = tgt["objects_bbox"]
            tgt_verbs_ids = tgt["verbs_id"]
            tgt_object_ids = tgt["objects_id"]
            tgt_interactions_id = self.objects_verbs_to_interaction_id[
                (tgt_object_ids, tgt_verbs_ids)
            ]

            # 1 means that the prediction is a true positive
            # 0 means that it is a false positive
            pred_tgt_matchings = torch.zeros(len(pred_scores), device=self.device)

            for pred_interaction_id in pred_interactions_id.unique():
                tgt_idxs = torch.nonzero(tgt_interactions_id == pred_interaction_id).squeeze(1)
                pred_idxs = torch.nonzero(pred_interactions_id == pred_interaction_id).squeeze(1)

                if len(tgt_idxs) == 0:
                    continue

                pred_tgt_matchings[pred_idxs] = self._match(
                    tgt_human_bboxes[tgt_idxs].view(-1, 4),
                    tgt_object_bboxes[tgt_idxs].view(-1, 4),
                    pred_human_bboxes[pred_idxs].view(-1, 4),
                    pred_object_bboxes[pred_idxs].view(-1, 4),
                    pred_scores[pred_idxs].view(-1),
                )

            all_pred_scores.append(pred_scores)
            all_pred_interactions_id.append(pred_interactions_id)
            all_pred_tgt_matchings.append(pred_tgt_matchings)

        all_pred_scores = (
            torch.cat(all_pred_scores) if all_pred_scores else torch.tensor([], device=self.device)
        )
        all_pred_interactions_id = (
            torch.cat(all_pred_interactions_id).long()
            if all_pred_interactions_id
            else torch.tensor([], device=self.device, dtype=torch.long)
        )
        all_pred_tgt_matchings = (
            torch.cat(all_pred_tgt_matchings)
            if all_pred_tgt_matchings
            else torch.tensor([], device=self.device, dtype=torch.float)
        )

        self.all_pred_scores.append(all_pred_scores)
        self.all_pred_interactions_id.append(all_pred_interactions_id)
        self.all_pred_tgt_matchings.append(all_pred_tgt_matchings)


    def compute(self, *args, **kwargs):
        if isinstance(self.all_pred_interactions_id, list):
            # It means this was executed in a single GPU
            all_pred_interactions_id = torch.cat(self.all_pred_interactions_id)
            all_pred_scores = torch.cat(self.all_pred_scores)
            all_pred_tgt_matchings = torch.cat(self.all_pred_tgt_matchings)
        else:
            all_pred_interactions_id = self.all_pred_interactions_id
            all_pred_scores = self.all_pred_scores
            all_pred_tgt_matchings = self.all_pred_tgt_matchings

        for int_id in all_pred_interactions_id.unique():
            samples_idx = torch.nonzero(all_pred_interactions_id == int_id).squeeze(1)
            self.output[int_id.item()] += all_pred_scores[samples_idx].tolist()
            self.labels[int_id.item()] += all_pred_tgt_matchings[samples_idx].tolist()

        output = [
            torch.tensor(self.output[idx], device=self.device, dtype=self.dtype)
            for idx in range(self.num_interactions)
        ]
        labels = [
            torch.tensor(self.labels[idx], device=self.device, dtype=self.dtype)
            for idx in range(self.num_interactions)
        ]

        self.aps, self.max_recs = self._compute_ap(output, labels)

        aps = {}
        aps["map"] = self.aps[self.interactions_id].mean()

        max_recs = {}
        max_recs["max-rec"] = self.max_recs[self.interactions_id].mean()

        return {**aps, **max_recs}

    def _match(
        self,
        tgt_human_bboxes,
        tgt_object_bboxes,
        pred_human_bboxes,
        pred_object_bboxes,
        pred_scores,
    ):
        iou_human = box_iou(tgt_human_bboxes, pred_human_bboxes)
        iou_object = box_iou(tgt_object_bboxes, pred_object_bboxes)
        iou = torch.min(iou_human, iou_object)

        max_iou, max_idx = iou.max(dim=0)

        if pred_scores is None:
            pred_scores = max_iou

        # Assign each detection to the best matching ground truth
        match = torch.zeros_like(iou)
        match[max_idx, torch.arange(iou.shape[1])] = max_iou

        # Threshold the matches
        match = match > self.iou_threshold

        pred_tgt_matching = torch.zeros_like(pred_scores)

        # Determine true positive
        for _, m in enumerate(match):
            match_idx = torch.nonzero(m).squeeze(1)
            if len(match_idx) == 0:
                continue

            match_scores = pred_scores[match_idx]
            pred_tgt_matching[match_idx[match_scores.argmax()]] = 1

        return pred_tgt_matching

    def _compute_ap(self, output, labels):
        ap = torch.zeros(len(output), device=self.device, dtype=output[0].dtype)
        max_rec = torch.zeros_like(ap)

        if self.ap_method == "11P":
            ap_method_fn = self._compute_per_class_ap_with_11_point_interpolation
        elif self.ap_method == "INT":
            ap_method_fn = self._compute_per_class_ap_with_interpolation
        else:
            ap_method_fn = self._compute_per_class_ap_as_auc

        if self.nproc < 2:
            for idx in range(len(output)):
                ap[idx], max_rec[idx] = self._compute_ap_for_each(
                    (
                        idx,
                        output[idx],
                        labels[idx],
                        self.num_annotations_per_interaction[idx],
                        ap_method_fn,
                    )
                )
        else:
            with torch.multiprocessing.get_context("spawn").Pool(self.nproc) as pool:
                for idx, results in enumerate(
                    pool.map(
                        self._compute_ap_for_each,
                        [
                            (
                                idx,
                                output[idx],
                                labels[idx],
                                self.num_annotations_per_interaction[idx],
                                ap_method_fn,
                            )
                            for idx in range(len(output))
                        ],
                    )
                ):
                    ap[idx], max_rec[idx] = results

        return ap, max_rec

    def _compute_ap_for_each(self, items):
        idx, output, labels, num_annotations, ap_method_fn = items

        if labels.sum() > num_annotations:
            log.warning(f"Class {idx}: number of true positives larger than that of ground truth")

            return 0, 0

        if len(output) and len(labels):
            prec, rec = self._compute_pr_for_each(output, labels, num_annotations)
            return ap_method_fn(prec, rec), rec[-1]
        else:
            return 0, 0

    def _compute_pr_for_each(self, output, labels, num_annotations):
        order = torch.argsort(output, descending=True)

        tp = labels[order]
        fp = 1 - tp
        tp = torch.cumsum(tp, dim=0)
        fp = torch.cumsum(fp, dim=0)

        prec = tp / (tp + fp)
        rec = tp / num_annotations

        # If nan, set to 0
        prec[torch.isnan(prec)] = 0
        rec[torch.isnan(rec)] = 0

        return prec, rec

    def _compute_per_class_ap_as_auc(self, prec, rec):
        ap = 0
        max_rec = rec[-1]

        for i in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[i] >= max_rec:
                break

            d_x = rec[i] - rec[i - 1]

            # Skip when negative example is registered
            if d_x == 0:
                continue

            ap += prec[i] * rec[i] if i == 0 else 0.5 * (prec[i] + prec[i - 1]) * d_x

        return ap

    def _compute_per_class_ap_with_11_point_interpolation(self, prec, rec):
        ap = 0
        for t in torch.linspace(0, 1, 11, dtype=prec.dtype, device=self.device):
            idxs = torch.nonzero(rec >= t).squeeze()
            if idxs.numel():
                ap += prec[idxs].max() / 11

        return ap

    def _compute_per_class_ap_with_interpolation(self, prec, rec):
        ap = 0
        max_rec = rec[-1]
        for idx in range(prec.numel()):
            # Stop when maximum recall is reached
            if rec[idx] >= max_rec:
                break

            d_x = rec[idx] - rec[idx - 1]

            # Skip when negative example is registered
            if d_x == 0:
                continue

            # Compute interpolated precision
            max_ = prec[idx:].max()
            ap += (
                max_ * rec[idx]
                if idx == 0
                else 0.5 * (max_ + torch.max(prec[idx - 1], max_)) * d_x
            )

        return ap
