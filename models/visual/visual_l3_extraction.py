"""
Contains the feature extraction part of a visual classifier based on the VGGish
pre-trained model. Handles the extraction and saving of the features.
"""
import os
import time

import openl3
import numpy as np
import pandas as pd
from skimage.io import imread

from utility.global_utilities import time_estimation, create_file_directory
from definitions import (IMAGE_TEST_DIR, IMAGE_TRAIN_DIR,
                         IMAGE_TRAIN_SPLIT_CSV, VISUAL_L3_TRAIN_FEATURES_FILE,
                         IMAGE_VAL_CSV, IMAGE_TEST_CSV,
                         VISUAL_L3_VAL_FEATURES_FILE,
                         VISUAL_L3_TEST_FEATURES_FILE,
                         VISUAL_L3_EMBED)


def extract_l3_image_features(csv_file: str, image_dir: str, file_out_path: str):
    """
    Extract feature data from images with OpenL3 pre-trained model.
    Write the data and labels as numpy arrays for later use.
    Expects all the images in csv_file to be in image_dir.

    Args:
        csv_file: a path of an input CSV file with columns 'image_name' and 'label'
        image_dir: a path of directory with the image files
        file_out_path: full path name of an output npz file
    """
    create_file_directory(file_out_path)

    model = openl3.models.load_image_embedding_model(input_repr='mel256',
                                                     content_type='env',
                                                     embedding_size=VISUAL_L3_EMBED)

    df = pd.read_csv(csv_file)
    ds_len = len(df)
    X = np.empty((ds_len, VISUAL_L3_EMBED), dtype=np.float32)
    y = np.empty(ds_len, dtype=np.int32)
    milestone = ds_len // 20
    start_time = time.time()
    for idx, row in df.iterrows():
        image_name = row['image_name']

        p = os.path.join(image_dir, image_name)
        image = imread(p)
        emb = openl3.get_image_embedding(image=image, model=model,
                                         verbose=False)

        X[idx] = emb
        y[idx] = int(row['label'])
        if idx % milestone == 0:
            time_estimation(start_time, idx+1, ds_len)

    np.savez_compressed(file_out_path, X=X, y=y)


if __name__ == '__main__':
    extract_train = True
    extract_val = True
    extract_test = True

    if extract_train:
        print(f'\nExtracting training features ...')
        extract_l3_image_features(csv_file=IMAGE_TRAIN_SPLIT_CSV,
                                  image_dir=IMAGE_TRAIN_DIR,
                                  file_out_path=VISUAL_L3_TRAIN_FEATURES_FILE)

    if extract_val:
        print(f'\nExtracting validation features ...')
        extract_l3_image_features(csv_file=IMAGE_VAL_CSV,
                                  image_dir=IMAGE_TRAIN_DIR,
                                  file_out_path=VISUAL_L3_VAL_FEATURES_FILE)

    if extract_test:
        print(f'\nExtracting testing features ...')
        extract_l3_image_features(csv_file=IMAGE_TEST_CSV,
                                  image_dir=IMAGE_TEST_DIR,
                                  file_out_path=VISUAL_L3_TEST_FEATURES_FILE)
