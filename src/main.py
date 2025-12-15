import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src.data.hoi_datamodule import HOIDataModule  # noqa: E402
from src.models.dysco import DYSCO  # noqa: E402
from src.trainer import Trainer  # noqa: E402
from src.utils import get_logger, init  # noqa: E402

log = get_logger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    log.info("Initializing...")
    init(cfg)

    log.info(f"Loading data <{cfg.data._target_}>")
    data: HOIDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Loading model <{cfg.model._target_}>")
    model: DYSCO = hydra.utils.instantiate(cfg.model)

    log.info(f"Loading trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Testing...")
    trainer.test(model=model, dataloader=data.get_test_loader())


if __name__ == "__main__":
    main()
