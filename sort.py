import os
import random
import shutil
import sys


def set_train_and_test_files(data_dir, percentage_to_train):
    # Directories for storing training and testing data
    train_path = 'train_path'  # Corrected from 'train_1000_10' to 'train_path'
    test_path = 'test_path'  # Use 'test_path' consistently

    print()

    # Create / empty the 'train_path' directory
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    else:
        remove_files(train_path)

    # Create / empty the 'test_path' directory
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    else:
        remove_files(test_path)

    print(
        f"\nRandomly sorting data from '{data_dir}' into '{train_path}' and '{test_path}' with {percentage_to_train} % going to train")
    print("'truth.dsv' file going into 'train_path'\n")

    train_ratio = percentage_to_train / 100

    files = os.listdir(data_dir)
    random.shuffle(files)

    train_file_count = int(len(files) * train_ratio)

    for idx, file_name in enumerate(files):
        src_path = os.path.join(data_dir, file_name)
        if file_name == 'truth.dsv':
            # Always copy truth.dsv to the train_path
            shutil.copy(src_path, os.path.join(train_path, file_name))
        elif idx < train_file_count:
            shutil.copy(src_path, os.path.join(train_path, file_name))
        else:
            shutil.copy(src_path, os.path.join(test_path, file_name))


def remove_files(directory):
    print(f"Removing all files in directory {directory}")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sort.py <data_directory> <percentage_to_train>")
        sys.exit(1)

    source_dir = sys.argv[1]
    percent_to_train = int(sys.argv[2])
    set_train_and_test_files(source_dir, percent_to_train)
