# Data-centric Machine Learning Challenge - MVEO 2023

A very simple baseline for data-centric machine learning challenge,
part of the Workshop on Machine Vision for Earth Observation (MVEO).

This repository uses TorchGeo and Lightning to train a U-Net to 
perform semantic segmentation on the DFC2022 dataset.

# Challenge

In this repo, there are two files (`train_coordinate_list.txt` and `val_coordinate_list.txt`) 
that list the patches used for training and validation, respectively.

Such files have the following organisation:
`<name of the city> <name of the image> <patch x coordinate> <patch y coordinate>`

Participants should propose techniques to select the training patches in order to improve the final outcome.
The final submission is a new `train_coordinate_list.txt` with the selected patches. 

# DFC2022 Dataset

This challenge is based on the [Data Fusion Contest 2022 dataset](https://www.grss-ieee.org/community/technical-committees/2022-ieee-grss-data-fusion-contest/).
You can download the dataset [here](https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022).
You will need to change the dataset path in the config.yaml file (datamodule.root_dir).

In this challenge, only the labelled part of the DFC2022 is being used.

# Dependencies

```
pip install -r requirements.txt
```

Torchgeo requires Python >= 3.10.

# Train

```python
python train.py --config_file config.yaml
```

# Code references

Other repos used as reference for this one:

- https://github.com/isaaccorley/dfc2022-baseline
- https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/dfc2022.py
