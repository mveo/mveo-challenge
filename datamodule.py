import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import kornia.augmentation as K
import lightning.pytorch as pl
from einops import rearrange

from matplotlib import colors
import matplotlib.pyplot as plt

from dfc2022 import DFC2022

from torchgeo.datasets.utils import percentile_normalization


DEFAULT_AUGS = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=["input", "mask"],
)


class DFC2022DataModule(pl.LightningDataModule):
    # Stats computed in labeled train set
    dem_min, dem_max = -79.18, 3020.26
    dem_nodata = -99999.0

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 8,
        num_workers: int = 0,
        train_coordinate_file_path: str = 'train_coords.txt',
        val_coordinate_file_path: str = 'val_coords.txt',
        patch_size: int = 512,
        predict_on: str = "val",
        augmentations=DEFAULT_AUGS,
        **kwargs,
    ):
        super().__init__()
        assert predict_on in DFC2022.metadata
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_coordinate_file_path = train_coordinate_file_path
        self.val_coordinate_file_path = val_coordinate_file_path
        self.patch_size = patch_size
        self.predict_on = predict_on
        self.augmentations = augmentations

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def preprocess(self, sample):
        # RGB is uint8 so divide by 255
        sample["image"][:3] /= 255.0
        sample["image"][-1] = (sample["image"][-1] - self.dem_min) / (
            self.dem_max - self.dem_min
        )
        sample["image"][-1] = torch.clip(sample["image"][-1], min=0.0, max=1.0)

        if "mask" in sample:
            # ignore the clouds and shadows class (not used in scoring)
            sample["mask"][sample["mask"] == 15] = 0
            sample["mask"] = rearrange(sample["mask"], "h w -> () h w")

        return sample

    def setup(self, stage=None):
        transforms = T.Compose([self.preprocess])
        test_transforms = T.Compose([self.preprocess])

        self.train_dataset = DFC2022(self.root_dir, self.train_coordinate_file_path, "train",
                                     self.patch_size, transforms=transforms)
        self.val_dataset = DFC2022(self.root_dir, self.val_coordinate_file_path, "train",
                                   self.patch_size, transforms=transforms)

        self.predict_dataset = DFC2022(self.root_dir, None, self.predict_on,
                                       self.patch_size, transforms=test_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            if self.augmentations is not None:
                batch["mask"] = batch["mask"].to(torch.float)
                batch["image"], batch["mask"] = self.augmentations(
                    batch["image"], batch["mask"]
                )
                batch["mask"] = batch["mask"].to(torch.long)

        batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch

    def plot(
            self,
            sample,
            show_titles=True,
            suptitle=None,
    ):
        ncols = 2
        image = sample["image"][:3]
        image = image.to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()

        dem = sample["image"][-1].numpy()
        dem = percentile_normalization(dem, lower=0, upper=100, axis=(0, 1))

        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        cmap = colors.ListedColormap(DFC2022.colormap)

        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(dem)
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(mask, cmap=cmap, interpolation="none")
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(pred, cmap=cmap, interpolation="none")
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(pred, cmap=cmap, interpolation="none")
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("DEM")

            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
