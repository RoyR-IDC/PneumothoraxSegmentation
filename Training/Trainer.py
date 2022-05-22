from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
from HW2.PneumothoraxSegmentationProject import WANTED_IMAGE_SIZE
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, dataset, model, loss_function, optimizer, hyperparams):
        print("entered Trainer class __init__ ")

        # save kwargs
        self._dataset = dataset
        self._model = model
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._hyperparams = hyperparams

        # create a SHUFFLING (!) dataloader
        self._dataloader = DataLoader(dataset,
                              batch_size=hyperparams['batch_size'],
                              num_workers=hyperparams['num_workers'],
                              shuffle=True)  # NOTE: shuffle = TRUE !!!

        # important: load model to cuda
        if hyperparams['device'].type == 'cuda':
            self._model = model.to(device=hyperparams['device'])

        # get num of batches
        self._save_number_of_batches()


    def train(self):
        print("****** begin training ******")
        num_of_epochs = self._hyperparams['num_of_epochs']
        loss_value_averages_of_all_epochs = []

        for iteration in range(self._hyperparams['num_of_epochs']):
            print(f'iteration {iteration + 1} ')

            # init variables for external loop
            dl_iter = iter(self._dataloader)  # iterator over the dataloader. called only once, outside of the loop, and from then on we use next() on that iterator
            loss_values_list = []

            for batch_index in range(self._num_of_batches):
                print(f'iteration {iteration+1} of {num_of_epochs} epochs: batch {batch_index+1} of {self._num_of_batches} batches') # "end='\r'" will cause the line to be overwritten the next print that comes

                # get current batch data
                data = next(dl_iter)  # note: "data" variable is a list with 3 elements
                '''
                Important notes:
                since `dataset[index] returns a dict, 
                that is also what the dataloader returns.
                it is apparently very smart, and adds up all inner items of the dict according to their type !`
                so now each key in the dict will contain `batch size` of items...
                so if batch size is 30, then:
                * data['Image'] is a tensor shaped (30,1024,1024)
                * data['Mask'] is a tensor shaped (30,1024,1024)
                '''

                x = data['Image']
                y = data['Mask']

                # load to device
                if self._hyperparams['device'].type == 'cuda':
                    x = x.to(device=self._hyperparams['device'])
                    y = y.to(device=self._hyperparams['device'])

                # Forward pass: compute predicted y by passing x to the model.
                result = self._model(x)

                # resize output mask to match size of prediction
                if (result.shape[-2] != y.shape[-2]) and (result.shape[-1] != y.shape[-1]):
                    transform = T.Compose([
                        T.Resize(WANTED_IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
                        T.ConvertImageDtype(torch.float),
                    ])
                    result = transform(result)

                # extract the prediction mask
                y_pred = result[:,1,:,:]  # TODO: not sure which channel is needed - 0 or 1. i'll take 1 for now
                y_true = y[:,0,:,:]  # only one channel here

                # Compute (and save) loss.
                loss = self._loss_function(y_pred, y_true)
                loss_values_list.append(loss.item())

                # Before the backward pass, use the optimizer object to zero all of the gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are accumulated in buffers( i.e, not overwritten)
                # whenever ".backward()" is called. Checkout docs of torch.autograd.backward for more details.
                self._optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its parameters
                self._optimizer.step()

                # delete unneeded tesnors from GPU to save space
                del x, y

            ##end of inner loop
            # print(f'\nfinished inner loop.')

            # data prints on the epoch that ended
            print(f'in this epoch: min loss {np.min(loss_values_list)} max loss {np.max(loss_values_list)}')
            average_value_this_epoch = np.mean(loss_values_list)
            print(f'in this epoch: average loss {average_value_this_epoch}')
            loss_value_averages_of_all_epochs.append(average_value_this_epoch)

        print(f'finished all epochs !                                         ')  # spaces ARE intended
        print(f'which means, that this model is now trained.')

        return self._model

    def _save_number_of_batches(self):
        # compute actual number of batches to train on in each epoch
        max_allowed_number_of_batches = self._hyperparams['max_allowed_number_of_batches']
        self._num_of_batches = (len(self._dataset) // self._dataloader.batch_size)
        if self._num_of_batches > max_allowed_number_of_batches:
            print(
                f'NOTE: in order to speed up training (while damaging accuracy) the number of batches per epoch was reduced from {self._num_of_batches} to {max_allowed_number_of_batches}')
            self._num_of_batches = max_allowed_number_of_batches
        else:
            # make sure there are no leftover datapoints not used because of "//"" calculation above
            if (len(self._dataset) % self._dataloader.batch_size) != 0:
                self._num_of_batches = self._num_of_batches + 1

