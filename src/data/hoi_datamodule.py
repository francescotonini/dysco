from typing import Optional

from torch.utils.data import DataLoader, Dataset

from src.utils.misc import default_collate_fn


class HOIDataModule:
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        dataset: Dataset,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = dataset

    def get_test_loader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=default_collate_fn,
            shuffle=False,
        )
