import numpy as np
import torch
from HW2.PneumothoraxSegmentationProject import TRAIN_RLE_ENCODINGS_FILEPATH, TRAIN_FILES_PATH
from HW2.PneumothoraxSegmentationProject.DataClasses.SIIMCustomDataset import SIIMDataset
from HW2.PneumothoraxSegmentationProject.Models.Unet import UNet
from HW2.PneumothoraxSegmentationProject.Training.Trainer import Trainer
from HW2.PneumothoraxSegmentationProject.Utilities.GPUUtilities import getAndPrintDeviceData_CUDAorCPU
from HW2.PneumothoraxSegmentationProject.Utilities.LossUtilities import MixedLoss
from HW2.PneumothoraxSegmentationProject.Utilities.PlotUtilities import plot_images, plot_prediction_comparisons


def get_hyperparams():
    hyperparameters = dict()
    hyperparameters['num_of_epochs'] = 2  # <-----------change to 5 at least
    hyperparameters['batch_size'] = 2
    hyperparameters['max_allowed_number_of_batches'] = 3  # <--------change to inf or 99999. anything below 1220 will cut some batches ... this is only used to speed up training
    hyperparameters['precent_of_dataset_allocated_for_training'] = 0.8  # TODO currently not used
    hyperparameters['learning_rate'] = 1e-4
    hyperparameters['momentum'] = 0.9
    hyperparameters['num_workers'] = 0  # <-------------- NOTE: default is 0, means everything happens serially.
    # see: https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
    hyperparameters['device'] = getAndPrintDeviceData_CUDAorCPU()
    return hyperparameters


def main():
    # creating the dataset takes about 4 mins
    dataset = SIIMDataset(dcm_files_path=TRAIN_FILES_PATH, rle_encodings_filepath=TRAIN_RLE_ENCODINGS_FILEPATH)
    dataset.export_to_csv(f'/Users/royrubin/Downloads/CustomSIIMDataset.csv')

    # plot a little bit of the dataset
    plot_images(dataset=dataset, indices=list(np.random.randint(low=20, high=500, size=15)))  #good indices: 132,

    # train model
    model = UNet()
    hyperparameters = get_hyperparams()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    loss_function = MixedLoss()
    trainer = Trainer(dataset=dataset,
                      model=model,
                      optimizer=optimizer,
                      loss_function=loss_function,
                      hyperparams=hyperparameters)
    model = trainer.train()

    # show some predictions
    plot_prediction_comparisons(dataset=dataset, model=model)


if __name__ == "__main__":
    main()
