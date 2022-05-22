import cv2
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from HW2.PneumothoraxSegmentationProject import WANTED_IMAGE_SIZE
from HW2.PneumothoraxSegmentationProject.Utilities.MaskUtilities import get_bounding_box


def show_patient_data(patient_file_name: str):
    pass
    # # displaying metadata
    # data = pydicom.dcmread(patient_file_name)
    # print(data)
    #
    # # displaying the image
    # img = pydicom.read_file(patient_file_name).pixel_array
    # plt.figure(0)
    # plt.grid(False)
    # plt.imshow(img, cmap='bone')


# def plot_imgs(uids_list, images_df, rle_encodings_df):
#     for uid in uids_list:
#         # prepare data for plots
#         try:
#             image = get_image_by_uid(images_df, uid)
#             mask = get_mask_by_uid(rle_encodings_df, image, uid)
#             # blob, blob_number = label(mask, return_num=True) # to talk with roi about this part
#             rmin, cmin, rmax, cmax = get_bounding_box(mask)
#         except Exception as e:
#             raise e
#             print(f'could not process image with uid {uid}.\nreason: {e}')
#             continue
#
#         # Create the figure
#         fig, axes = plt.subplots(1, 4, figsize=(20, 15))
#
#         # Create the plot for the original image
#         axes[0].imshow(image, cmap='bone')
#         axes[0].set_title('Original Image')
#
#         # Create the plot for the Pneumathorax mask
#         axes[1].imshow(mask, cmap='gray')
#         axes[1].set_title('Mask Only')
#
#         # Create the plot for the original image with the mask on top of it
#         axes[2].imshow(image, cmap='bone')
#         axes[2].imshow(mask, alpha=0.3, cmap='Reds')
#         axes[2].set_title('Image + mask')
#
#         # Create the plot for original image +mask+ mask bounding box
#         cv2.rectangle(image, (cmin, rmin), (cmax, rmax), (255, 255, 0), 5)
#         axes[3].imshow(image)
#         axes[3].imshow(mask, alpha=0.3, cmap='Reds')
#         axes[3].set_title('Image + Mask + Bounding Box')
#
#         # Finally, show image
#         plt.show()


def plot_images(dataset, indices: list):
    for index in indices:
        # prepare data for plots
        try:
            record = dataset[index]
            image = record['Image'][0, :, :].detach().numpy()
            mask = record['Mask'][0, :, :].detach().numpy()
            uid = record['UID']
            top_left, bottom_right = get_bounding_box(mask)
        except Exception as e:
            raise e

        # Create the figure
        fig, axes = plt.subplots(1, 3, figsize=(10, 8))
        plt.suptitle(f'Plotting UID {uid}')

        # Create the plot for the original image
        axes[0].imshow(image, cmap='bone')
        axes[0].set_title(f'Original Image (index {index})')

        # Create the plot for the Pneumathorax mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask Only')

        # Create the plot for original image +mask+ mask bounding box
        if top_left is not None and bottom_right is not None:
            cv2.rectangle(mask, top_left, bottom_right, (255, 255, 0), 5)
        axes[2].imshow(image, cmap='bone')
        axes[2].imshow(mask, alpha=0.3, cmap='Reds')
        axes[2].set_title('Image + Mask + Bounding Box')

        # Finally, show image
        plt.show()


def plot_prediction_comparisons(dataset, model):
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)  # NOTE: shuffle = TRUE !!!
    dl_iter = iter(dataloader)

    for batch_index in range(10):

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
        uid = data['UID']

        # Forward pass: compute predicted y by passing x to the model.
        result = model(x)

        # resize output mask to match size of prediction
        if (result.shape[-2] != y.shape[-2]) and (result.shape[-1] != y.shape[-1]):
            transform = T.Compose([
                T.Resize(WANTED_IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
                T.ConvertImageDtype(torch.float),
            ])
            result = transform(result)

        # extract the prediction mask
        orig_img = x[0, 0, :, :].detach().numpy()
        y_pred = result[0, 1, :, :].detach().numpy()  # TODO: not sure which channel is needed - 0 or 1. i'll take 1 for now
        y_true = y[0, 0, :, :].detach().numpy()  # only one channel here

        # Create the figure
        fig, axes = plt.subplots(1, 3, figsize=(10, 8))
        plt.suptitle(f'Plotting UID {uid}')

        # Create the plot for the original image
        axes[0].imshow(orig_img, cmap='bone')
        axes[0].set_title(f'Original Image')

        # Create the plot for the Pneumathorax mask
        axes[1].imshow(orig_img, cmap='bone')
        axes[1].imshow(y_true, alpha=0.3, cmap='Reds')
        axes[1].set_title('y_true Mask on original image')

        # Create the plot for original image +mask+ mask bounding box
        axes[2].imshow(orig_img, cmap='bone')
        axes[2].imshow(y_pred, alpha=0.3, cmap='Reds')
        axes[2].set_title('y_true Mask on original image')

        # Finally, show image
        plt.show()

