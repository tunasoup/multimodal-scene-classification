"""
Contains the feature extraction part of a visual classifier based on the ResNet50
pre-trained model. Handles the extraction and saving of the features.
"""
import os
import time

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from utility.global_utilities import time_estimation, create_file_directory
from definitions import (IMAGE_TEST_DIR, IMAGE_TRAIN_DIR,
                         IMAGE_TRAIN_SPLIT_CSV,
                         VISUAL_RESNET_TRAIN_FEATURES_FILE,
                         VISUAL_RESNET_VAL_FEATURES_FILE,
                         VISUAL_RESNET_TEST_FEATURES_FILE,
                         IMAGE_VAL_CSV, IMAGE_TEST_CSV, VISUAL_RESNET_EMBED)


class ImageData(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform: transforms.Compose):
        """
        Generate a torch Dataset from images. Applies transforming required
        by a pre-trained network.

        Meant to be used for a pre-trained network.

        Args:
            csv_file: a path to a csv file with data of the images and
                      column 'image_name'
            image_dir: a path of a directory with the input images
            transform: a torchvision Compose object with the required
                       pre-trained network transformations
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.df.label[idx]
        img_p = os.path.join(self.image_dir, self.df.image_name[idx])
        sample = cv.imread(img_p)

        sample = self.transform(sample)

        return sample, label


def pretrain(csv_file: str, image_dir: str, file_out_path: str, device: str):
    """
    Feed images into a pre-trained network and write feature vectors to a file

    Args:
        csv_file: a path to a csv file with data of the images and column 'image_name'
        image_dir: a path to a directory with the input images
        file_out_path:  a path of an output npz file
        device: device to run pytorch on
    """
    create_file_directory(file_out_path)

    resnet = resnet50(pretrained=True)  # Installs weights on the first use
    pre_model = nn.Sequential(*(list(resnet.children())[:-1]))
    pre_model.to(device)
    pre_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  # (HxWxC, [0,256] -> CxHxW, [0.0,1.0]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_set = ImageData(csv_file, image_dir, transform)
    ds_len = len(data_set)
    X = torch.empty(ds_len, VISUAL_RESNET_EMBED, device=device)
    y = torch.empty(ds_len, dtype=torch.long, device=device)

    loader = DataLoader(data_set, batch_size=32)

    counter = 0
    start_time = time.time()
    for batch_idx, data in enumerate(loader):
        inputs = data[0].contiguous().float()
        labels = data[1].contiguous()

        b_len = len(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)
        y[counter:counter+b_len] = labels

        with torch.no_grad():
            outputs = pre_model(inputs)
            outputs = torch.squeeze(outputs, dim=2)
            outputs = torch.squeeze(outputs, dim=2)
            X[counter:counter+b_len] = outputs

        counter += b_len
        time_estimation(start_time, counter, ds_len)

    # Write the features
    np.savez_compressed(file_out_path,
                        X=X.detach().cpu().numpy(),
                        y=y.detach().cpu().numpy())
    print(f'Pretrained network outputs written in {file_out_path}\n')


if __name__ == '__main__':
    extract_train = True
    extract_val = True
    extract_test = True

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    if extract_train:
        print(f'\nExtracting training features ...')
        pretrain(IMAGE_TRAIN_SPLIT_CSV, image_dir=IMAGE_TRAIN_DIR,
                 file_out_path=VISUAL_RESNET_TRAIN_FEATURES_FILE,
                 device=device)

    if extract_val:
        print(f'\nExtracting validation features ...')
        pretrain(IMAGE_VAL_CSV, image_dir=IMAGE_TRAIN_DIR,
                 file_out_path=VISUAL_RESNET_VAL_FEATURES_FILE,
                 device=device)

    if extract_test:
        print(f'\nExtracting testing features ...')
        pretrain(IMAGE_TEST_CSV, image_dir=IMAGE_TEST_DIR,
                 file_out_path=VISUAL_RESNET_TEST_FEATURES_FILE,
                 device=device)
