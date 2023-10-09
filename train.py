import os
import argparse

# import pytorch_lightning as pl
import lightning.pytorch as pl
from omegaconf import OmegaConf

from torchgeo.trainers import SemanticSegmentationTask
from datamodule import DFC2022DataModule


def main(_config):
    pl.seed_everything(0)

    task = SemanticSegmentationTask(**_config.learning)  # model = unet (with imagenet), loss = ce (no class weights)
    datamodule = DFC2022DataModule(**_config.datamodule)
    trainer = pl.Trainer(**_config.trainer)
    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to config.yaml file")
    parser.add_argument("--config_file", type=str, required=True, help="Path to config.yaml file")
    args = parser.parse_args()

    _config = OmegaConf.load(args.config_file)

    main(_config)
