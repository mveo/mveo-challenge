import glob
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from rasterio.enums import Resampling


class DFC2022(Dataset):
    classes = [
        "No information",
        "Urban fabric",
        "Industrial, commercial, public, military, private and transport units",
        "Mine, dump and construction sites",
        "Artificial non-agricultural vegetated areas",
        "Arable land (annual crops)",
        "Permanent crops",
        "Pastures",
        "Complex and mixed cultivation patterns",
        "Orchards at the fringe of urban classes",
        "Forests",
        "Herbaceous vegetation associations",
        "Open spaces with little or no vegetation",
        "Wetlands",
        "Water",
        "Clouds and Shadows",
    ]

    colormap = [
        "#231F20",
        "#DB5F57",
        "#DB9757",
        "#DBD057",
        "#ADDB57",
        "#75DB57",
        "#7BC47B",
        "#58B158",
        "#D4F6D4",
        "#B0E2B0",
        "#008000",
        "#58B0A7",
        "#995D13",
        "#579BDB",
        "#0062FF",
        "#231F20",
    ]

    metadata = {
        "train": "labeled_train",
        "val": "val",
        "test": "test",
    }

    image_root = "BDORTHO"
    dem_root = "RGEALTI"
    target_root = "UrbanAtlas"

    def __init__(
            self,
            root,
            coordinate_file_path=None,
            split="train",
            patch_size=256,
            transforms=None
    ) -> None:
        assert split in self.metadata
        self.root = root
        self.coordinate_file_path = coordinate_file_path
        self.split = split
        self.patch_size = patch_size
        self.transforms = transforms

        self.files = self._load_files()

    def __getitem__(self, index):
        img_path, dem_path = self.files[index]["image"], self.files[index]["dem"]
        if self.split == "train":
            label_path = self.files[index]["target"]
        if self.coordinate_file_path is not None:
            coord_x, coord_y = self.files[index]['coord_x'], self.files[index]['coord_y']

        image = self._load_image(img_path)
        if '44-2013-0375-6697-LA93-0M50-E080' in img_path:
            print('check -------------------------------->>>>>>>>>>', image.shape)
        dem = self._load_image(dem_path, shape=image.shape[1:])
        image = torch.cat(tensors=[image, dem], dim=0)

        if self.split == "train":
            mask = self._load_target(label_path)

        if self.coordinate_file_path is not None:
            image = image[:, coord_x:coord_x + self.patch_size, coord_y:coord_y + self.patch_size]
            mask = mask[coord_x:coord_x + self.patch_size, coord_y:coord_y + self.patch_size]

        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)

    def _load_files(self):
        if self.coordinate_file_path is not None:
            # read list of patches
            file_list = np.loadtxt(self.coordinate_file_path, dtype=str, delimiter=' ')

            files = []
            for x in file_list:
                img_path = os.path.join(self.root, self.metadata[self.split], x[0], self.image_root, x[1] + ".tif")
                dem_path = os.path.join(self.root, self.metadata[self.split], x[0], self.dem_root,
                                        x[1] + '_RGEALTI.tif')
                if self.split == "train":
                    label_path = os.path.join(self.root, self.metadata[self.split],
                                              x[0], self.target_root, x[1] + '_UA2012.tif')
                    files.append(dict(image=img_path, dem=dem_path, target=label_path,
                                      coord_x=int(x[2]), coord_y=int(x[3])))
                else:
                    files.append(dict(image=img_path, dem=dem_path, coord_x=int(x[2]), coord_y=int(x[3])))
        else:
            directory = os.path.join(self.root, self.metadata[self.split])
            images = glob.glob(os.path.join(directory, "**", self.image_root, "*.tif"), recursive=True)

            files = []
            for image in sorted(images):
                dem = image.replace(self.image_root, self.dem_root)
                dem = f"{os.path.splitext(dem)[0]}_RGEALTI.tif"

                if self.split == "train":
                    target = image.replace(self.image_root, self.target_root)
                    target = f"{os.path.splitext(target)[0]}_UA2012.tif"
                    files.append(dict(image=image, dem=dem, target=target))
                else:
                    files.append(dict(image=image, dem=dem))

        return files

    def _load_image(self, path, shape=None):
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
            )
            tensor = torch.from_numpy(array)
            return tensor

    def _load_target(self, path):
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.int_]" = f.read(
                indexes=1, out_dtype="int32", resampling=Resampling.bilinear
            )
            tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.long)
            return tensor

