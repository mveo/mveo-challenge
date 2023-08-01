# Data-centric Machine Learning Challenge - MVEO 2023

This repo has some resources for the data-centric machine learning challenge,
part of the Workshop on Machine Vision for Earth Observation (MVEO).

This repository uses TorchGeo and Lightning to train a U-Net to 
perform semantic segmentation on the DFC2022 dataset.

## Challenge Summary

Participants of this challenge are asked to develop a ranking strategy that assigns a 
score to each sample from a pool of candidate examples based on each sample's importance 
to the training procedure. The generated score/ranking will then be used to select a core 
set of training samples to train a pre-defined classifier, in this case a U-Net. 
Success is measured by training the aforementioned classifier multiple times using 
training datasets of different sizes based on the given ranking/scores (for example, 
by training a model using the top 1000 and the top 500 samples) and calculating the 
average Jaccard index using an undisclosed test dataset for the trained models. 

Find more information about the challenge [here](https://mveo.github.io/challenge.html).

## DFC2022 Dataset

This challenge is based on the [Data Fusion Contest 2022 dataset](https://www.grss-ieee.org/community/technical-committees/2022-ieee-grss-data-fusion-contest/).
You can download the dataset [here](https://ieee-dataport.org/competitions/data-fusion-contest-2022-dfc2022).
You will need to change the dataset path in the config.yaml file (datamodule.root_dir).

## Pool of Candidate Samples for Training

In this repo, the file `train_coordinate_list.txt` has the pool of candidate samples.
Each row of this file represents a candidate sample for the training and is composed of 5 columns:

`< name of the city > < name of the image > < patch x coordinate > < patch y coordinate > < sample score (float from 0.0 to 1.0) >`

Currently, all samples have the same score/importance (i.e., <em>1.0</em>). 
As mentioned before, the main objective of the participants is to develop a 
ranking system that assigns different scores (from 0.0 to 1.0) to the 
candidate samples depending on their importance to the training procedure.

## Validation set

This repo also has the file `val_image_list.txt` which list the images that should be used for validation.

## Submission

The final submission must be a file with all candidate samples, 
each with the same 5 columns, but varying the score depending on 
the importance of the example. This file must be sent to 
[data-centric-challenge-mveo@googlegroups.com](data-centric-challenge-mveo@googlegroups.com), with title 
[MVEO Challenge Submission TEAMNAME], where TEAMNAME 
should be the name of your team for the [leaderboard](https://mveo.github.io/challenge.html). 

## Dependencies

```
pip install -r requirements.txt
```

Torchgeo requires Python >= 3.10.

## Train

```python
python train.py --config_file config.yaml
```

## Code references

Other repos used as reference for this one:

- https://github.com/isaaccorley/dfc2022-baseline
- https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/dfc2022.py
