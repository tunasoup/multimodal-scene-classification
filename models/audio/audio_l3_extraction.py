"""
Contains the feature extraction part of an audio classifier based on the OpenL3
pre-trained model. Handles the extraction and saving of the features.
"""
import os
import time

import openl3
import soundfile as sf
import numpy as np
import pandas as pd

from definitions import (AUDIO_L3_TRAIN_FEATURES_FILE,
                         AUDIO_L3_VAL_FEATURES_FILE,
                         AUDIO_L3_TEST_FEATURES_FILE,
                         TRAIN_SPLIT_CSV, VAL_CSV, TEST_CSV,
                         TAU_AUDIO_DIR, VECTORS_PER_AUDIO,
                         AUDIO_L3_EMBED)

from utility.global_utilities import time_estimation, create_file_directory


def extract_l3_audio_features(csv_file: str, audio_dir: str, file_out_path: str):
    """
    Extract feature data from images with OpenL3 pre-trained model.
    Write the data and labels as numpy arrays for later use.
    Expects all the audio in csv_file to be in audio_dir.

    Args:
        csv_file: a path of an input CSV file with columns 'audio_name' and 'label'
        audio_dir: a path of a directory with the audio files
        file_out_path: a path of an output npz file
    """
    create_file_directory(file_out_path)

    model = openl3.models.load_audio_embedding_model(input_repr='mel256',
                                                     content_type='env',
                                                     embedding_size=AUDIO_L3_EMBED)

    df = pd.read_csv(csv_file)
    ds_len = len(df)

    X = np.empty((ds_len * VECTORS_PER_AUDIO, AUDIO_L3_EMBED), dtype=np.float32)
    y = np.empty(ds_len * VECTORS_PER_AUDIO, dtype=np.int32)
    milestone = ds_len // 20
    counter = 0
    start_time = time.time()
    for idx, row in df.iterrows():
        audio_name = row['audio_name']

        p = os.path.join(audio_dir, audio_name)
        audio, sr = sf.read(p)
        if len(audio) > 480000:
            audio = audio[:480000]
        emb, ts = openl3.get_audio_embedding(audio=audio, sr=sr, model=model,
                                             hop_size=1, center=False,
                                             verbose=False)

        dim = emb.shape[0]
        X[counter:counter+dim] = emb
        y[counter:counter+dim] = int(row['label'])
        counter += dim
        if idx % milestone == 0:
            time_estimation(start_time, idx + 1, ds_len)

    np.savez_compressed(file_out_path, X=X, y=y)


if __name__ == '__main__':
    extract_train = True
    extract_val = True
    extract_test = True

    if extract_train:
        print(f'\nExtracting training features ...')
        extract_l3_audio_features(csv_file=TRAIN_SPLIT_CSV,
                                  audio_dir=TAU_AUDIO_DIR,
                                  file_out_path=AUDIO_L3_TRAIN_FEATURES_FILE)

    if extract_val:
        print(f'\nExtracting validation features ...')
        extract_l3_audio_features(csv_file=VAL_CSV,
                                  audio_dir=TAU_AUDIO_DIR,
                                  file_out_path=AUDIO_L3_VAL_FEATURES_FILE)

    if extract_test:
        print(f'\nExtracting testing features ...')
        extract_l3_audio_features(csv_file=TEST_CSV,
                                  audio_dir=TAU_AUDIO_DIR,
                                  file_out_path=AUDIO_L3_TEST_FEATURES_FILE)

