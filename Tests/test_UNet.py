import unittest
import torch
from HW2.PneumothoraxSegmentationProject import TRAIN_FILES_PATH, TRAIN_RLE_ENCODINGS_FILEPATH
from HW2.PneumothoraxSegmentationProject.DataClasses.SIIMCustomDataset import SIIMDataset
from HW2.PneumothoraxSegmentationProject.Models.Unet import UNet
from HW2.PneumothoraxSegmentationProject.Utilities.PlotUtilities import plot_images


class TestUNet(unittest.TestCase):
    def test_prediction(self):
        image = torch.rand((2, 1, 572, 572))  # single greyscale image
        model = UNet()
        res = model(image)
        print(res.size())

    def test_paper_output_sizes(self):
        image = torch.rand((1, 1, 572, 572))  # single greyscale image
        model = UNet()
        res = model(image)
        assert list(res.size()) == [1, 2, 388, 388]
