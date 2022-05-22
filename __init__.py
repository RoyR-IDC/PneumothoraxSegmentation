import os

DEBUG_MODE = True  # True \ False

USER = 'roy'   # 'roy' \ 'or'

# general folder path
if USER == 'roy':
    # I downloaded the data from: 'https://www.kaggle.com/datasets/seesee/siim-train-test'
    FOLDER_PATH = '/Users/royrubin/Downloads/siim/'
else:
    FOLDER_PATH = 'C:\\MSC\\HC\\HW2\\siim\\'

# paths for files
TRAIN_RLE_ENCODINGS_FILEPATH = os.path.join(FOLDER_PATH, 'train-rle.csv')
TRAIN_FILES_PATH = os.path.join(FOLDER_PATH, 'dicom-images-train/')
TEST_FILES_PATH = os.path.join(FOLDER_PATH, 'dicom-images-test/')


FAST_LOAD_MODE = False        # True / False
FAST_LOAD_NUM_OF_SAMPLES = 500
WANTED_IMAGE_SIZE = 1024      # 572 works with baseline UNET

SELECTIVE_LOAD_MODE = True    # True / False
SUBSET_IDS_PATH = '/Users/royrubin/PycharmProjects/helfcare/HW2/train-rle4000.csv'
