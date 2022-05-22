# Data handling imports
import pandas as pd

from HW2.PneumothoraxSegmentationProject import USER

pd.reset_option('max_colwidth')
import numpy as np

# Read DICOM format
import pydicom

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# General purpose imports
import os
from glob import glob

# Benchmark \ Baseline U-net
# pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp

def get_():
    # general folder path
    if USER == 'roy':
        # I downloaded the data from: 'https://www.kaggle.com/datasets/seesee/siim-train-test'
        folder_path = '/Users/royrubin/Downloads/siim/'
    else:
        folder_path = '... OR - your path'

    # paths for files
    train_rle_encodings_file_path = os.path.join(folder_path, 'train-rle.csv')
    train_files_path = os.path.join(folder_path, 'dicom-images-train/')
    test_files_path = os.path.join(folder_path, 'dicom-images-test/')

    print(f'train_rle_encodings_file_path: {train_rle_encodings_file_path}')
    print(f'train_files_path: {train_files_path}')
    print(f'test_files_path: {test_files_path}')

    # reading all dcm files into train and text
    train_file_names = sorted(glob(train_files_path + "*/*/*.dcm"))
    test_file_names = sorted(glob(test_files_path + "*/*/*.dcm"))


    print(f"\ntrain files: amount {len(train_file_names)}, examples: \n{train_file_names[0]}\n{train_file_names[1]}")
    print(f"\ntest files: amount {len(test_file_names)}, examples: \n{test_file_names[0]}\n{test_file_names[1]}")

    train_rle_encodings_df = pd.read_csv(train_rle_encodings_file_path, delimiter=",")
    train_rle_encodings_df.rename(columns={" EncodedPixels": "EncodedPixels"}, inplace=True)

    print(f'size {train_rle_encodings_df.shape}')

    # around 3 mins
    temp = [{'UID': pydicom.read_file(file_name).SOPInstanceUID, 'Image': pydicom.read_file(file_name).pixel_array}
            for file_name in train_file_names]
    train_images_df = pd.DataFrame(temp)

    print(f'size {len(train_images_df)}')
    print(f'shape of single image {train_images_df.iloc[0].Image.shape}')

    # around 30 secs
    train_metadata = [pydicom.dcmread(file_name) for file_name in train_file_names]

    print(type(train_metadata[0]))

    result = {
        'a': 1,
        'b': 2,
        'c': 3,
    }

    return result
