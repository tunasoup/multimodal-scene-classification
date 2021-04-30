"""
VGGish feature extraction for audio.
"""

from __future__ import print_function
import os
import time

import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd

from . import vggish_params, vggish_slim
from . import vggish_input
from definitions import (VGGISH_MODEL, VECTORS_PER_AUDIO,
                         AUDIO_VGGISH_EMBED)
from utilities.global_utilities import time_estimation, create_file_directory


def extract_features(csv_file: str, audio_dir: str, file_out_path: str):
    """
    Run audio files from AUDIO_DIR through vggish and write the
    extracted features to an npz file

    Args:
        csv_file: a path to an input CSV file with columns 'audio_name' and
                 'label'
        audio_dir: a path to a directory of the audio files
        file_out_path: a path to an output npz file

    """
    create_file_directory(file_out_path)

    audio_segments = round(VECTORS_PER_AUDIO / 0.96)
    df = pd.read_csv(csv_file)
    ds_len = len(df)
    X = np.empty((ds_len*audio_segments, AUDIO_VGGISH_EMBED), dtype=np.float32)
    y = np.empty(ds_len*audio_segments, dtype=np.int32)
    milestone = ds_len // 20
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, VGGISH_MODEL)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Read files in a csv
        counter = 0
        start_time = time.time()
        for idx, row in df.iterrows():
            audio_name = row['audio_name']

            wav_file = os.path.join(audio_dir, audio_name)
            logmel = vggish_input.wavfile_to_examples(wav_file)

            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={
                                             features_tensor: logmel})

            dim = embedding_batch.shape[0]
            X[counter:counter+dim] = embedding_batch
            y[counter:counter+dim] = int(row['label'])
            counter += dim

            # Print progress
            if idx % milestone == 0:
                time_estimation(start_time, idx + 1, ds_len)

    # Write the features
    np.savez_compressed(file_out_path, X=X, y=y)
