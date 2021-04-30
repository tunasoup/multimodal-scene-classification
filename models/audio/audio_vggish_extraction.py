"""
Contains the feature extraction part of an audio classifier based on the VGGish
pre-trained model. Handles the extraction and saving of the features.
"""
import multiprocessing

from models.audio.vggish import extract_features
from definitions import (TEST_CSV, TRAIN_SPLIT_CSV, AUDIO_VGGISH_TRAIN_FEATURES_FILE,
                         VAL_CSV, AUDIO_VGGISH_VAL_FEATURES_FILE,
                         AUDIO_VGGISH_TEST_FEATURES_FILE,
                         TAU_AUDIO_DIR)


def extract_process(csv_path: str, audio_dir: str, feature_file: str):
    """
    Extract features from audio files and save them in npz format.
    Run as a process to release the VRAM used by tensorflow after the
    function finishes.

    Args:
        csv_path: a path of an input CSV file with columns 'audio-name' and 'label'
        audio_dir: a path of a directory with the audio files
        feature_file: a path of an output npz file

    """
    process_extract = multiprocessing.Process(target=extract_features,
                                              args=(csv_path,
                                                    audio_dir,
                                                    feature_file))
    process_extract.start()
    process_extract.join()


if __name__ == '__main__':
    extract_train = True
    extract_val = True
    extract_test = True

    if extract_train:
        print(f'\nExtracting training features ...')
        extract_process(TRAIN_SPLIT_CSV, TAU_AUDIO_DIR,
                        AUDIO_VGGISH_TRAIN_FEATURES_FILE)

    if extract_val:
        print(f'\nExtracting validation features ...')
        extract_process(VAL_CSV, TAU_AUDIO_DIR, AUDIO_VGGISH_VAL_FEATURES_FILE)

    if extract_test:
        print(f'\nExtracting testing features ...')
        extract_process(TEST_CSV, TAU_AUDIO_DIR, AUDIO_VGGISH_TEST_FEATURES_FILE)