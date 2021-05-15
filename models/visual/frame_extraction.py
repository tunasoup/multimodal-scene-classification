"""
Responsible of extracting frames/images from videos. The frames are resized
and saved to a local directory. Corresponding CSV files are also created.
"""
import os
import time

import cv2 as cv
import pandas as pd

from utility.csv_handling import generate_image_csv_from_dir_csv
from utility.global_utilities import time_estimation
from definitions import (IMAGE_TRAIN_DIR, IMAGE_TEST_DIR,
                         IMAGE_VAL_DIR,
                         FRAMES_PER_VIDEO, TAU_VIDEO_DIR,
                         TRAIN_SPLIT_CSV, VAL_CSV, TEST_CSV,
                         IMAGE_TRAIN_SPLIT_CSV, IMAGE_VAL_CSV, IMAGE_TEST_CSV)


def extract_videos_on_csv(csv_file: str, video_dir: str, out_dir: str,
                          frames: int = FRAMES_PER_VIDEO):
    """
    Extract frames from a video so they can be input to a pre-trained image
    model. Files are read from a csv and all of the files are expected to be in the
    given directory. The frames are resized and saved locally.

    Args:
        csv_file: a path to a csv with a column 'video_name'
        video_dir: a path to a directory consisting of all the videos on the csv
        out_dir: a path to a directory where the frame images are saved
        frames: amount of frames extracted from a video
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(csv_file)
    file_count = len(df)
    print(f'Preprocessing {file_count} videos...')
    milestone = file_count // 20

    already_preprocessed_videos = os.listdir(out_dir)
    already_preprocessed_videos = [x.split('_frame')[0] for x in already_preprocessed_videos[::frames]]

    start_time = time.time()
    for idx, file_name in enumerate(df.video_name):
        vid = os.path.join(video_dir, file_name)
        name = file_name.split('.')[0]

        # Ignore if video already preprocessed, assumes same frame count was used
        if name in already_preprocessed_videos:
            if idx % milestone == 0:
                time_estimation(start_time, idx + 1, file_count)
            continue

        cap = cv.VideoCapture(vid)
        counter = 1
        total_frames = cap.get(7)
        n = total_frames // frames

        while cap.isOpened():
            success, img = cap.read()
            frame_id = cap.get(1)
            if not success:
                break

            # Extract every nth frame
            if frame_id % n == 0:
                img = cv.resize(img, (244, 244))
                path_out = os.path.join(out_dir, f'{name}_frame{counter}.png')
                cv.imwrite(path_out, img)
                counter += 1
        cap.release()

        if idx % milestone == 0:
            time_estimation(start_time, idx + 1, file_count)
    print(f'Preprocessing done.\n')


def extract_single_video(video_name: str, video_dir: str, out_dir: str,
                         frames: int = FRAMES_PER_VIDEO):
    """
    Extract frames from a single video. The frames are resized and saved locally.

    Args:
        video_name: name of the video
        video_dir: a path to a directory of the video
        out_dir: a path to a directory of the output images
        frames: number of frames extracted from an image
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    vid = os.path.join(video_dir, video_name)
    name = video_name.split('.')[0]
    cap = cv.VideoCapture(vid)
    total_frames = cap.get(7)
    n = total_frames // frames
    counter = 1

    while cap.isOpened():
        success, img = cap.read()
        frame_id = cap.get(1)
        if not success:
            break

        # Extract every nth frame
        if frame_id % n == 0:
            img = cv.resize(img, (244, 244))
            path_out = os.path.join(out_dir, f'{name}_frame{counter}.png')
            cv.imwrite(path_out, img)
            counter += 1
    cap.release()


if __name__ == '__main__':
    preprocess_train = True
    preprocess_val = True
    preprocess_test = True

    if preprocess_train:
        extract_videos_on_csv(TRAIN_SPLIT_CSV, TAU_VIDEO_DIR, IMAGE_TRAIN_DIR)
        generate_image_csv_from_dir_csv(csv_in=TRAIN_SPLIT_CSV,
                                        csv_out=IMAGE_TRAIN_SPLIT_CSV,
                                        image_dir=IMAGE_TRAIN_DIR)

    if preprocess_val:
        extract_videos_on_csv(VAL_CSV, TAU_VIDEO_DIR, IMAGE_VAL_DIR)
        generate_image_csv_from_dir_csv(csv_in=VAL_CSV,
                                        csv_out=IMAGE_VAL_CSV,
                                        image_dir=IMAGE_TRAIN_DIR)

    if preprocess_test:
        extract_videos_on_csv(TEST_CSV, TAU_VIDEO_DIR, IMAGE_TEST_DIR)
        generate_image_csv_from_dir_csv(csv_in=TEST_CSV,
                                        csv_out=IMAGE_TEST_CSV,
                                        image_dir=IMAGE_TEST_DIR)

