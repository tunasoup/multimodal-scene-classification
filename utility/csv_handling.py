"""
Contains different CSV functions. The data splitting of training and validation
is also done here. TAU_CSV files refer to training and testing csv files
that come with the development set.
"""
import csv
import os
import shutil
from typing import Dict, List
import random

import pandas as pd

from definitions import (TAU_CSV_DIR, TAU_AUDIO_DIR, TAU_VIDEO_DIR,
                         TEST_CSV, LABELS, TRAIN_CSV, VAL_CSV,
                         TRAIN_SPLIT_CSV, SEED)
from utility.global_utilities import create_file_directory

random.seed(SEED)

FILE_TRAIN = 'fold1_train.csv'                   # TAU training data
FILE_TEST = 'fold1_evaluate.csv'                 # TAU testing data


def copy_files_from_tau_csv(file_name: str, audio_out_dir: str,
                            video_out_dir: str, n: int):
    """
    For an unsorted and original csv, get the audio and video names of files
    with certain intervals, and copy them to another directory.

    Args:
        file_name: csv file name, located in TAU_CSV_DIR
        audio_out_dir: directory for the audio
        video_out_dir: directory for the video
        n: copy every nth file
    """
    csv_in = os.path.join(TAU_CSV_DIR, file_name)
    with open(csv_in, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        counter = n
        for row in reader:
            # Take every nth occurence
            counter += 1
            if counter >= n:
                counter = 0
                text = list(row.values())[0]
                t = text.replace('\t', '/').split('/')
                audio_name = t[1]
                video_name = t[3]
                shutil.copy(os.path.join(TAU_AUDIO_DIR, audio_name),
                            audio_out_dir)
                shutil.copy(os.path.join(TAU_VIDEO_DIR, video_name),
                            video_out_dir)
                print(audio_name + '\t' + video_name)


def improve_csv(file_in_name: str, file_out_path: str, n: int = 1):
    """
    Copy data from an original CSV, and generate a better column structure.

    Args:
        file_in_name: a name of an original CSV, located in TAU_CSV_DIR
        file_out_path: a path to a new or overwritable CSV
        n: copy every nth file
    """
    if n < 1:
        return

    create_file_directory(file_out_path)

    print(f'Copying and improving every {n}th row from {file_in_name} to '
          f'{file_out_path}')

    csv_in = os.path.join(TAU_CSV_DIR, file_in_name)
    df_in = pd.read_csv(csv_in, sep='\t')
    field_names = ['audio_name', 'video_name', 'scene', 'label', 'city',
                   'location', 'file_name']
    df_out = pd.DataFrame(columns=field_names)
    for i, row in enumerate(df_in.sort_values(by=['filename']).values):
        if i % n == 0:
            data = row[0]
            file_name = data.replace('.', '/').split('/')[1]
            audio_name = file_name + '.wav'
            video_name = file_name + '.mp4'
            data2 = file_name.split('-')
            scene = data2[0]
            city = data2[1]
            location = data2[2]
            label = LABELS[scene]

            df = pd.DataFrame(
                [[audio_name, video_name, scene, label, city, location,
                  file_name]],
                columns=field_names)
            df_out = df_out.append(df)

    df_out.to_csv(file_out_path, index=False)
    print(f'CSV copied.\n')


def generate_image_csv_from_dir(image_dir: str, csv_out: str):
    """
    Generate a csv from a directory with images, where all the images are used.

    Args:
        image_dir: a path to a directory with the images
        csv_out: a path to the new csv
    """
    create_file_directory(csv_out)

    print(f'Generating {csv_out} from {image_dir} ...')
    field_names = ['image_name', 'scene', 'label', 'city', 'location',
                   'file_name']
    df = pd.DataFrame(columns=field_names)
    for image_name in os.listdir(image_dir):
        data = image_name.split('-')
        scene = data[0]
        label = LABELS[scene]
        city = data[1]
        location = data[2]
        data = image_name.split('_frame')
        file_name = data[0]

        df2 = pd.DataFrame([[image_name, scene, label, city, location, file_name]],
                           columns=field_names)
        df = df.append(df2)

    df.to_csv(csv_out, index=False)
    print(f'Generated {csv_out}.\n')


def generate_image_csv_from_dir_csv(csv_in: str, image_dir: str, csv_out: str):
    """
    Generate a csv from a directory with images, but only with the files
    appearing on the given csv file.

    Args:
        csv_in: a path to a csv file with a column 'file_name'
        image_dir: a path to a directory with the images
        csv_out: a path to the new csv
    """
    create_file_directory(csv_out)

    print(f'Generating {csv_out} with {image_dir} ...')
    field_names = ['image_name', 'scene', 'label', 'city', 'location',
                   'file_name']
    all_images = set(os.listdir(image_dir))
    df_out = pd.DataFrame(columns=field_names)
    df_in = pd.read_csv(csv_in)
    for row in df_in.sort_values(by=['file_name']).values:
        file_name = row[6]
        i = 1
        while True:
            image_name = f'{file_name}_frame{i}.png'

            if image_name in all_images:
                scene = row[2]
                label = row[3]
                city = row[4]
                location = row[5]
                df2 = pd.DataFrame(
                    [[image_name, scene, label, city, location, file_name]],
                    columns=field_names)
                df_out = df_out.append(df2)
                i += 1

            else:
                break

    df_out.to_csv(csv_out, index=False)
    print(f'Generated {csv_out}.\n')


def generate_csv_from_dir(directory: str, csv_out: str):
    """
    Generate a csv from a dictionary, regarding information about audio
    and video names.

    Args:
        directory: a path to a directory of files with the name format:
                   'scene_name-city-locationID-recordID-*.*'
        csv_out: a path to the new csv
    """
    create_file_directory(csv_out)

    print(f'Generating {csv_out} from {directory} ...')
    field_names = ['audio_name', 'video_name', 'scene', 'label',
                   'city', 'location', 'file_name']
    df = pd.DataFrame(columns=field_names)
    for file_name in os.listdir(directory):
        data = file_name.split('-')
        scene = data[0]
        label = LABELS[scene]
        city = data[1]
        location = data[2]
        data = file_name.split('.')
        name = data[0]
        audio_name = f'{name}.wav'
        video_name = f'{name}.mp4'

        df2 = pd.DataFrame([[audio_name, video_name, scene, label, city,
                            location, name]],
                           columns=field_names)
        df = df.append(df2)

    df.to_csv(csv_out, index=False)
    print(f'Generated {csv_out}.')


def move_files_on_csv(csv_file: str, column: str, in_dir: str, out_dir: str):
    """
    Move the files found on the given csv file to a different directory.

    Args:
        csv_file: a path to a csv file
        column: a csv column to match a file with (e.g. audio_name)
        in_dir: a path to a directory where the files are moved from
        out_dir: a path to a directory where the files are moved to
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    counter = 0
    df = pd.read_csv(csv_file)
    for x in df[column]:
        file = os.path.join(in_dir, x)
        try:
            shutil.move(file, out_dir)
            counter += 1
        except FileNotFoundError:
            print(f'File {x} not found, skipping')

    print(f'Moved {counter} files from {in_dir} to {out_dir}')


def split_data(ratio: float, csv_file: str, train_csv_out: str, val_csv_out: str):
    """
    Split data to training and validation sets with even labels according to
    the given ratio. One location id only appears in one set.

    Args:
        ratio: the fraction used as validation data (0.0 - 1.0)
        csv_file: a path to csv file with all the data with columns 'label' and
                  'location'
        train_csv_out: a path to a csv file which will list the training files
        val_csv_out: a path to a csv file which will list the validation files

    Returns:
        False if the ratio is not between 0 and 1, otherwise True
    """
    if ratio >= 1:
        print('ratio too high')
        return False
    elif ratio <= 0:
        print('ratio too low')
        return False

    df = pd.read_csv(csv_file)
    df_train = pd.DataFrame(columns=df.columns)
    df_val = pd.DataFrame(columns=df.columns)

    # get all labels
    labels = [x for _, x in df.groupby('label')]

    # list all locations for every label
    for i in range(len(labels)):
        loc_ids = [x for _, x in labels[i].groupby('location')]
        count = 0
        index_ids = []
        a = list(range(len(loc_ids)))

        # randomly add same location instances in a bulk until limit reached
        while True:
            b = random.choice(a)
            index_ids.append(b)
            a.remove(b)
            count += len(loc_ids[b])
            df_train = df_train.append(loc_ids[b])
            if count >= int((1 - ratio) * len(labels[i])):
                break

        # add the rest of the locations to validation
        for j in range(len(a)):
            df_val = df_val.append(loc_ids[a[j]])

    # save csvs
    df_train.to_csv(train_csv_out, index=False)
    df_val.to_csv(val_csv_out, index=False)
    return True


def split_data_to_n(n: int, csv_file: str, csv_dir_out: str):
    """
    Splits the data in the given csv file to n portions. The portions
    are written in separate CSV files in the given output directory.

    Currently outputs an uneven final split.

    Args:
        n: number of splits
        csv_file: a path to a csv file with columns 'label' and 'location'
        csv_dir_out: a path to a directory for the csv file outputs, should
                     not contain anything else

    Returns:
        False if not the number of splits is too small
    """
    if n < 2:
        print('Not enough splits')
        return False

    df = pd.read_csv(csv_file)

    # get all labels
    labels = [x for _, x in df.groupby('label')]

    csvs = []
    for m in range(n):
        df_i = pd.DataFrame(columns=df.columns)
        csvs.append(df_i)

    # list all locations for every label
    for i in range(len(labels)):
        loc_ids = [x for _, x in labels[i].groupby('location')]
        index_ids = []
        a = list(range(len(loc_ids)))

        # TODO: create an algorithm that splits locations groups as evenly as
        #  possible to n proportions
        # randomly add same location instances in a bulk until limit reached
        for idx, df_i in enumerate(csvs[:-1]):
            count = 0
            while True:
                b = random.choice(a)
                index_ids.append(b)
                a.remove(b)
                count += len(loc_ids[b])
                df_i = df_i.append(loc_ids[b])
                if count >= int(1/n * len(labels[i])):
                    csvs[idx] = df_i
                    break

        # add the rest of the locations to validation
        for j in range(len(a)):
            csvs[-1] = csvs[-1].append(loc_ids[a[j]])

    # save csvs
    for idx, df_i in enumerate(csvs):
        name = f'train_split_{idx}.csv'
        p = os.path.join(csv_dir_out, name)
        df_i.to_csv(p, index=False)
    return True


class SceneData:
    def __init__(self, city: str, loc_id: int):
        """
        Holds miscellaneous data for a scene.

        Args:
            city: name of a city
            loc_id: location identifier for a recording
        """
        self.count = 0
        self.cities: Dict[str, int] = {city: 1}                          # city: occurence
        self.city_locs: Dict[str, Dict[int, int]] = {city: {loc_id: 1}}  # city: loc_id: occurence

    def add_city_occ(self, city: str):
        if city in self.cities:
            self.cities[city] += 1
        else:
            self.cities[city] = 1

    def add_loc_occ(self, city: str, loc_id: int):
        if city in self.city_locs:
            if loc_id in self.city_locs[city]:
                self.city_locs[city][loc_id] += 1
            else:
                self.city_locs[city][loc_id] = 1
        else:
            self.city_locs[city] = {loc_id: 1}


def get_scene_data(csv_in: str):
    """
    Obtain scene data from the given csv.

    Args:
        csv_in: a path to a csv file with the columns 'scene', 'city', 'location'

    Returns:
        a dictionary with a scene as the key and SceneData as value
    """
    occurences: Dict[str, SceneData] = {}
    with open(csv_in, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            scene = row['scene']
            city = row['city']
            loc_id = int(row['location'])

            if scene in occurences:
                occurences[scene].add_city_occ(city)
                occurences[scene].add_loc_occ(city, loc_id)
            else:
                occurences[scene] = SceneData(city, loc_id)

            occurences[scene].count += 1

    return occurences


def print_data(occurences, file_name: str):
    """
    Print miscellaneous data about the given occurences.

    Args:
        occurences:
        file_name: a name or a path of the file, which is only used for status print
    """
    print(f'\nData regarding {file_name}:\n')
    file_amount = sum(map(lambda x: x.count, occurences.values()))
    all_cities = set()
    all_loc_ids = set()

    for key in sorted(occurences):
        scene = key
        scene_obj = occurences[key]
        amount = scene_obj.count
        city_amount = len(scene_obj.cities)
        loc_amount = sum(len(v) for v in scene_obj.city_locs.values())

        print(f'{scene} & {amount} & {city_amount} & {loc_amount} & {amount / 6}')  # latex table

        print(f'\n{scene}:\n'
              f'\tTotal amount: {amount}\n'
              f'\tTotal cities: {city_amount}\n'
              f'\tCities: {sorted(scene_obj.city_locs)}\n'
              f'\tTotal locations: {loc_amount}\n'
              f'\tCity data:'
              )

        for city in sorted(scene_obj.city_locs):
            all_cities.add(city)
            city_obj = scene_obj.city_locs[city]
            print(f'\t\t{city}:\n'
                  f'\t\t\t# of locations: {len(city_obj)}')
            for loc_id in sorted(city_obj):
                all_loc_ids.add(loc_id)
                print(f'\t\t\t\t{loc_id}: {city_obj[loc_id]}')

    print(f'Conclusion:\n'
          f'Files: {file_amount}\n'
          f'Cities: {len(all_cities)}\n'
          f'\t {sorted(all_cities)}\n'
          f'Locations: {len(all_loc_ids)}')


if __name__ == '__main__':
    print('Running csv_handling.py')
    generate_csvs = True
    do_split = True
    print_data_info = False

    if generate_csvs:
        improve_csv(file_in_name=FILE_TRAIN, file_out_path=TRAIN_CSV)
        improve_csv(file_in_name=FILE_TEST, file_out_path=TEST_CSV)

    if do_split:
        split_data(ratio=0.2, csv_file=TRAIN_CSV,
                   train_csv_out=TRAIN_SPLIT_CSV,
                   val_csv_out=VAL_CSV)

    if print_data_info:
        print("Printing data")
        for file in [TRAIN_CSV, TEST_CSV]:
            data = get_scene_data(file)
            print_data(data, file)
