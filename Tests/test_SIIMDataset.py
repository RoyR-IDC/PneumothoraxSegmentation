import unittest
import torch
from HW2.PneumothoraxSegmentationProject import TRAIN_FILES_PATH, TRAIN_RLE_ENCODINGS_FILEPATH
from HW2.PneumothoraxSegmentationProject.DataClasses.SIIMCustomDataset import SIIMDataset
from HW2.PneumothoraxSegmentationProject.Models.Unet import UNet
from HW2.PneumothoraxSegmentationProject.Utilities.PlotUtilities import plot_images


class TestUNet(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self._dataset = SIIMDataset(dcm_files_path=TRAIN_FILES_PATH, rle_encodings_filepath=TRAIN_RLE_ENCODINGS_FILEPATH)

    def test_dataset_len(self):
        expected_dataset_size = 12954
        self.assertEqual(len(self._dataset), expected_dataset_size)

    def test_get_item(self):
        row = self._dataset[0]
        pass  #TODO: test the result here
