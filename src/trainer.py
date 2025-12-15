from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import lightning as L
import torch
import torchmetrics
import torchmetrics.detection
import torchmetrics.detection.mean_ap
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from prettytable import PrettyTable
from tqdm import tqdm

from src.utils import get_logger

log = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        output_dir: str,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        limit_batches: Union[int, float] = float("inf"),
        save_preds: bool = False,
    ) -> None:
        self._fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )

        assert Path(output_dir).exists(), f"Output directory {output_dir} does not exist."

        self._limit_batches: Union[int, float] = limit_batches
        self._dataloader: Optional[torch.utils.data.DataLoader] = None
        self._model: Optional[L.LightningModule] = None
        self._output_dir: Optional[Path] = Path(output_dir)
        self._save_preds: bool = save_preds

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def fabric(self) -> L.Fabric:
        return self._fabric

    @property
    def limit_batches(self) -> Union[int, float]:
        return self._limit_batches

    @property
    def dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        return self._dataloader

    @property
    def dataset(self) -> Optional[torch.utils.data.Dataset]:
        return self.dataloader.dataset if self.dataloader is not None else None

    @property
    def model(self) -> Optional[L.LightningModule]:
        return self._model

    def test(
        self,
        model: L.LightningModule,
        dataloader: torch.utils.data.DataLoader,
    ):
        # Stop if no dataloader is passed
        if dataloader is None:
            return

        # Stop if no test_step is implemented
        if dataloader is not None and not is_overridden("test_step", _unwrap_objects(model)):
            log.error(
                "Your LightningModule does not have a test_step implemented, "
                "but you passed a test dataloader. Skipping Testing."
            )
            return

        # Launch Fabric
        self._fabric.launch()

        # Setup dataset and dataloader
        dataloader.dataset.setup(
            limit_samples=(
                self._limit_batches * dataloader.batch_size if self._limit_batches else None
            )
        )
        self._dataloader = self.fabric.setup_dataloaders(dataloader, move_to_device=True)

        # Setup model
        self._model = self.fabric.setup(model, move_to_device=True)
        self._model.trainer = self  # NOTE: this could cause a circular reference
        self._model.setup(stage="test")

        # Print number of parameters if available
        if hasattr(self._model, "num_parameters") and self._model.num_parameters:
            total_params = 0
            for name, num_params in self._model.num_parameters.items():
                total_params += num_params
                num_million_params = num_params // 1_000_000

                log.info(f"Number of parameters in {name}: {num_million_params}M")
            log.info(f"Total number of parameters: {total_params // 1_000_000}M")

        # Run!
        self.test_loop()

        # Log metrics
        self.log_metrics(model.metrics)

        # Close logger
        if self.fabric.loggers:
            self.fabric.logger.save()

        # Print final message with output directory
        log.info(f"Done! Experiment run dir is {self._output_dir}")

    def test_loop(
        self,
    ):
        self._model.eval()
        torch.set_grad_enabled(False)

        self.fabric.call("on_test_epoch_start")
        iterable = self._progbar_wrapper(
            self._dataloader,
            total=(
                min(len(self._dataloader), self.limit_batches)
                if self.limit_batches
                else len(self._dataloader)
            ),
            desc="Test",
        )

        for batch_idx, batch in enumerate(iterable):
            if self.limit_batches and batch_idx >= self.limit_batches:
                break

            self.fabric.call("on_test_batch_start", batch, batch_idx)
            out = self._model.test_step(batch, batch_idx)
            self.fabric.call("on_test_batch_end", out, batch, batch_idx)

            if self._save_preds:
                self.save_artifacts(out, f"test_preds_{batch_idx}.pth")
                self.save_artifacts(batch, f"test_batch_{batch_idx}.pth")

            self._format_iterable(
                iterable, None, "test"
            )  # If you want to add metrics, pass them as the second argument

        self.fabric.call("on_test_epoch_end")

    def save_output(
        self, content: Any, filename: str, subfolder: Optional[Path] = None
    ):
        filename = f"{self.fabric.global_rank}_{filename}"

        if subfolder:
            output_dir = self._output_dir / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / filename
        torch.save(content, save_path)
        log.debug(f"Saved {filename} to {save_path}")

    @property
    def artifacts_dir(self) -> Path:
        path = self._output_dir / "artifacts"
        if not path.exists():
            path.mkdir(parents=True)

        return self._output_dir / "artifacts"

    def save_artifacts(
        self, content: Any, filename: str, subfolder: Optional[Path] = None
    ):
        save_dir = Path("artifacts")

        if subfolder is not None:
            save_dir = save_dir / subfolder

        self.save_output(content, filename, subfolder=save_dir)

    def _progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        if self._fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)

        return iterable

    @staticmethod
    def _format_iterable(
        prog_bar,
        candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]],
        prefix: str,
    ):
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)

    def log_metrics(self, metrics: dict[str, torchmetrics.Metric]):
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]

        metrics_scores = {}
        for name, metric in metrics.items():
            metric_score = metric.compute()

            # If metric score is a dict, print all values
            if isinstance(metric_score, dict):
                for k, v in metric_score.items():
                    if isinstance(v, torch.Tensor):
                        # v could be a single value tensor or a tensor with multiple values.
                        # In the latter case, we ignore it
                        if v.numel() == 1:
                            v = v.item()
                        else:
                            continue

                    metrics_scores[f"{name}_{k}"] = v
                    table.add_row([f"{name}_{k}", f"{v:.4f}"])
            elif isinstance(metric_score, torch.Tensor) and metric_score.numel() != 1:
                log.warning(f"Metric {name} returned a tensor with more than one value. Ignoring.")
            else:
                metrics_scores[name] = metric_score
                table.add_row([name, f"{metric_score:.4f}"])

        self.fabric.log_dict(metrics_scores)

        print(table)
