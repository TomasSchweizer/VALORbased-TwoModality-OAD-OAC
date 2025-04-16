# VALOR-based Two-Modality Training for Online Action Detection (OAD) and Online Action Captioning (OAC)

This repository holds the code replicate the results of the VALOR-based Two-Modality Training.
It can be used to train and test in parallel, and also run the recurrent inference for OAD and OAC.

## Setup

The workspace is setup using devcontainers. Therefore the devcontainer extension is needed for visual studio code.
The devcontainer is then setting up the development environment.
The devcontainer is based on the image: pytorch:2.1.2-cuda12.1-cudnn8-devel
Then via pip the requirements.txt packages are installed, most importantly this code base uses:
- wandb (For expirement logging. Here you have to add your own login in the file postCreateCommand.sh)
- hydracore (For configuration management.)
- pytorch-lightning (For training and testing.)
Also the mmaction2 code based is git cloned into this repository and all necessary dependencies are installed.

## Pre-trained checkpoints

For training from scratch the following checkpoints are needed.
For the text encoder and two-modality decoder either:
- BERT_B checkpoint: bert-base-uncased.bin, bert-base-uncased-vocab.txt
or:
- VALOR_L BERT_B checkpoint: bert_b_valor_l.pt, bert-base-uncased-vocab.txt

## Data

For training on the BaseballPitch subset of the THUMOS'14 dataset:
- We need the pre-extracted features for each video belonging to the BaseballPitch subset in form of numpy files.
- and the onehot labels which are also stored for each video separately in numpy files.
- Then we need the label-based segment-level captions for the BaseballPitch subset, here the choice is between:
    - simple label-based captions
    - ChatGPT-4o-mini captions
    - human-created captions
The needed data info is defined in the following config: thumosbaseball_data-featurescaptions_fps-24_frames-16_swstride-6.yaml

For training on the whole THUMOS'14 dataset:
- The pre-extracted features for each video again in the form of numpy files.
- and the onehot labels which are also stored for each video separately in numpy files.
- Then we need the label-based segment-level captions, here the choice is between:
    - simple label-based captions
    - ChatGPT-4o-mini captions
The needed data info is defined in the following config: thumos_data-featurescaptions_fps-24_frames-16_swstride-6.yaml

For testing on either the BaseballPitch subset or the whole THUMOS'14 dataset:
- Only the pre-extracted features for each video and the onehot labels which are also stored for each video separately in numpy files are needed.

## Parallel Training and Validation/Testin

To train we have to launch the train.py script in the src directory.
This than loads the train.yaml config file.
In this train.yaml config file only the experiment config is of interest.
The expirement config selects:
- callbacks (e.g. learning rate and checkpoint saving) 
- datamodule (wrapping the dataloaders)
- module (wrapping the models, defining optimizers)
- model (defining the chosen model architecture)
- checkpoint_path (if specified the training continues from the given checkpoint)
- trainer (epochs, strategy=ddp, which gpu to use, check_val_every_n_epoch, ...)
- env (defining the random seed for random, numpy, torch)
- logger (defining the wandb experiment/run, !change group to sort experiments after what is tested!)

The most important configs are datamodule and model:

The datamodule confing defines:
- the dataset_info meaning which kind of data is loaded for the train and valid dataset.
- datasets: The train dataset and the test dataset
- loaders: config for the train and valid dataloader.
The moduel config defines:
- data_info (copies important properties of dataset for the model dimensions)
- All components of the model:
    - video_decoder
    - shared_text_multimodal_encoder_decoder
    - contrastive_heads
- weight initalization 
- criterions (all losses defined to train the model)
- metrics (all computed metrics split in train and test metrics)
- optimizer
- scheduler


## Recurrent Testing

For recurrent testing the test.py script in the src directory has to be launched.
This uses the test.yaml config file, in which the a test experiment has to be called.
Then the rest of the configs is the same as for training. Just that you have to give a checkpoint of a trained model to the experiment config field checkpoint_path.