"""
Utility functions that do not contain 3rd party libraries.

Can be used by both environments.
"""
import os
import time


def time_estimation(start_time: time, current_files: int, total_files: int):
    """
    Estimate the time to process the remaining files and print it.

    Args:
        start_time: start time of the process
        current_files: number of processed files
        total_files: total number of files to process
    """
    if current_files == 0:
        return
    current_time = time.time()
    duration = (current_time - start_time)
    time_left = duration / current_files * total_files - duration
    minutes = time_left // 60
    seconds = time_left % 60
    print(f'{100 * current_files / total_files:.1f}% done '
          f'({current_files}/{total_files}), '
          f'estimated time: {minutes:.0f}m {seconds:.0f}s')


def create_file_directory(file_path: str):
    """
    Create a directory for the given file if it does not exist yet.

    Args:
        file_path: a file's path, which contains the directory
    """
    dir_path = os.path.dirname(file_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
