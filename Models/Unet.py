'''
taken from https://www.youtube.com/watch?v=u1loyDCoGbE&ab_channel=AbhishekThakur
'''

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        print(f'Entered UNet `__init__` method')
        super(UNet, self).__init__()

        # Initialize the max pooling layers for the contracting section of UNet
        self._max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Initialize the double convolutions needed for the contracting section of UNet
        self._double_convolution_1 = self._get_double_convolution(input_channels=1, output_channels=64)
        self._double_convolution_2 = self._get_double_convolution(input_channels=64, output_channels=128)
        self._double_convolution_3 = self._get_double_convolution(input_channels=128, output_channels=256)
        self._double_convolution_4 = self._get_double_convolution(input_channels=256, output_channels=512)
        self._double_convolution_5 = self._get_double_convolution(input_channels=512, output_channels=1024)

        # Initialize the up convolutions needed in the expanding section of UNet
        self._up_convolution_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2)  #  "... upsampling of the feature map ..."
        self._up_convolution_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2)  #  "... upsampling of the feature map ..."
        self._up_convolution_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2)  #  "... upsampling of the feature map ..."
        self._up_convolution_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2)  #  "... upsampling of the feature map ..."

        # Initialize the double convolutions needed for the expanding section of UNet
        self._double_convolution_6 = self._get_double_convolution(input_channels=1024, output_channels=512)
        self._double_convolution_7 = self._get_double_convolution(input_channels=512, output_channels=256)
        self._double_convolution_8 = self._get_double_convolution(input_channels=256, output_channels=128)
        self._double_convolution_9 = self._get_double_convolution(input_channels=128, output_channels=64)

        # output layer
        self._out = nn.Conv2d(
            in_channels=64,
            out_channels=2,   #  <----- !!! this represents "number of classes" - in the segmentation, or out of the segmentation !
            kernel_size=(1, 1),
        )

        # useful variables init
        self._data_for_concatanation = None

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        # print(f'Entered `encode`')

        x1 = self._double_convolution_1(X)
        x1_pool = self._max_pool_2x2(x1)           # move down 1
        x2 = self._double_convolution_2(x1_pool)
        x2_pool = self._max_pool_2x2(x2)           # move down 2
        x3 = self._double_convolution_3(x2_pool)
        x3_pool = self._max_pool_2x2(x3)           # move down 3
        x4 = self._double_convolution_4(x3_pool)
        x4_pool = self._max_pool_2x2(x4)           # move down 4
        x5 = self._double_convolution_5(x4_pool)

        # print(f'x size {X.size()}')
        # print(f'x1 size {x1.size()}')
        # print(f'x5 size {x5.size()}')

        self._data_for_concatanation = {
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4,
        }

        result = x5
        return result

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        # print(f'Entered `decode`')

        ### STEP 1 ###
        move_up_1 = self._up_convolution_1(X)
        cropped_x4 = self._crop_image(initial_tensor=self._data_for_concatanation['x4'],
                                      target_height_and_width=move_up_1.size()[2])
        concatenated_1 = torch.cat([move_up_1, cropped_x4], 1)  # maybe concatanation should be HAFOOCHA ?
        x6 = self._double_convolution_6(concatenated_1)

        # print(f'move_up_1 size {move_up_1.size()}')
        # print(f'cropped_x4 size {cropped_x4.size()}')
        # print(f'concatanated size {concatenated_1.size()}')
        # print(f'x6 size {x6.size()}')

        ### STEP 2 ###
        move_up_2 = self._up_convolution_2(x6)
        cropped_x3 = self._crop_image(initial_tensor=self._data_for_concatanation['x3'],
                                      target_height_and_width=move_up_2.size()[2])
        concatenated_2 = torch.cat([move_up_2, cropped_x3], 1)  # maybe concatanation should be HAFOOCHA ?
        x7 = self._double_convolution_7(concatenated_2)

        ### STEP 3 ###
        move_up_3 = self._up_convolution_3(x7)
        cropped_x2 = self._crop_image(initial_tensor=self._data_for_concatanation['x2'],
                                      target_height_and_width=move_up_3.size()[2])
        concatenated_3 = torch.cat([move_up_3, cropped_x2], 1)  # maybe concatanation should be HAFOOCHA ?
        x8 = self._double_convolution_8(concatenated_3)

        ### STEP 4 ###
        move_up_4 = self._up_convolution_4(x8)
        cropped_x1 = self._crop_image(initial_tensor=self._data_for_concatanation['x1'],
                                      target_height_and_width=move_up_4.size()[2])
        concatenated_4 = torch.cat([move_up_4, cropped_x1], 1)  # maybe concatanation should be HAFOOCHA ?
        x9 = self._double_convolution_9(concatenated_4)

        # print(f'move_up_4 size {move_up_4.size()}')
        # print(f'cropped_x1 size {cropped_x1.size()}')
        # print(f'concatenated_4 size {concatenated_4.size()}')
        # print(f'x9 size {x9.size()}')

        ### FINAL OPERATION ###
        result = self._out(x9)
        return result

    def forward(self, X: torch.Tensor):
        """

        :param X: a torch tensor of the input images
        :return:
        """
        # print(f'Entered `forward`')
        # print(f'Received: type {type(X)} size {X.size()}')
        encoded = self.encode(X)
        decoded = self.decode(encoded)
        return decoded

    def backward(self):
        pass

    def predict(self):
        pass

    """Assisting functions"""

    def _get_double_convolution(self, input_channels, output_channels):
        """
        in the UNet, each level (before up/down sampling) has 2 convolutions in it
        returns 2 convolutions.

        :param input_channels:
        :param output_channels:
        :return:
        """
        convolution_double_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=(3, 3), padding=0),
            nn.ReLU(inplace=True),
        )
        return convolution_double_layer

    def _crop_image(self, initial_tensor: torch.Tensor, target_height_and_width: int):
        initial_size = initial_tensor.size()[2]
        delta = (initial_size - target_height_and_width) // 2
        cropped_tensor = initial_tensor[:, :, delta:initial_size - delta, delta:initial_size - delta]  # why ?
        return cropped_tensor


