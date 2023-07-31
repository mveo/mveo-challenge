import argparse
import glob
import os

import numpy as np
import kornia.augmentation as K
import rasterio
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from einops import rearrange

from torchgeo.trainers import SemanticSegmentationTask
from datamodule import DFC2022DataModule


def write_mask(mask, path, output_dir):
    with rasterio.open(path) as src:
        profile = src.profile
    profile["count"] = 1
    profile["dtype"] = "uint8"
    region = os.path.dirname(path).split(os.sep)[-2]
    filename = os.path.basename(os.path.splitext(path)[0])
    output_path = os.path.join(output_dir, region, f"{filename}_prediction.tif")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)


@torch.no_grad()
def main(config_file, log_dir, device):
    general_config = OmegaConf.load(config_file)

    # Load checkpoint and config
    trained_params = OmegaConf.load(os.path.join(log_dir, "hparams.yaml"))
    ckpt = glob.glob(os.path.join(log_dir, "checkpoints", "*.ckpt"))[0]

    # Load model
    task = SemanticSegmentationTask.load_from_checkpoint(ckpt)
    task = task.to(device)
    task.eval()

    # Load datamodule and dataloader
    datamodule = DFC2022DataModule(**general_config.datamodule)
    datamodule.setup()
    dataloader = datamodule.test_dataloader()

    pad = K.PadTo(size=(2048, 2048), pad_mode="constant", pad_value=0.0)

    accuracy = []
    jaccard = []

    accuracy_metric = MulticlassAccuracy(num_classes=trained_params.num_classes,
                                         ignore_index=trained_params.ignore_index,
                                         multidim_average="global", average="micro").to(device)
    jaccard_metric = MulticlassJaccardIndex(num_classes=trained_params.num_classes,
                                            ignore_index=trained_params.ignore_index, average="micro").to(device)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = batch["image"].to(device)
        h, w = x.shape[-2:]
        x = pad(x)
        preds = task(x)
        preds = preds[0, :, :h, :w]
        preds = rearrange(preds, "c h w -> (h w) c")
        target = batch["mask"].to(device).flatten()
        print(preds.get_device(), target.get_device())

        accuracy.append(accuracy_metric(preds, target).item())
        jaccard.append(jaccard_metric(preds, target).item())

    print(np.mean(accuracy), np.std(accuracy))
    print(np.mean(jaccard), np.std(jaccard))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to config.yaml file")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to log directory containing config.yaml and checkpoint")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args.config_file, args.log_dir, args.device)
