import numpy as np
import torch

from HW2.PneumothoraxSegmentationProject import WANTED_IMAGE_SIZE


def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def get_mask_from_rle_encodings(rle_encodings, img_width, img_height):
    # create the mask
    # note: there can be more than one RLE encoding per image

    if isinstance(rle_encodings, str):
        print(f'this is a problem ....')

    elif isinstance(rle_encodings, list):
        if rle_encodings == ["-1"] or rle_encodings == [" -1"]:
            empty_mask = np.zeros([WANTED_IMAGE_SIZE, WANTED_IMAGE_SIZE])
            return empty_mask
        final_mask = None
        for rle_encoding in rle_encodings:
            assert isinstance(rle_encoding, str)
            current_mask = rle2mask(rle=rle_encoding, width=img_width, height=img_height)
            if final_mask is None:
                final_mask = current_mask
            else:
                # print(f'another mask is added')
                final_mask += current_mask  # Important logic

        final_mask[final_mask > 0] = 255
        mask = final_mask
        # mask needs to be rotated to fit the original image
        mask = np.rot90(mask, 3)  # rotating three times 90 to the right place
        mask = np.flip(mask, axis=1)
        return mask
    else:
        print(f'unexpected input')
        empty_mask = np.zeros([WANTED_IMAGE_SIZE, WANTED_IMAGE_SIZE])
        return empty_mask


def get_bounding_box(image: np.ndarray):
    # # return max and min of a mask to draw bounding box
    x, y = np.where(image)
    if len(x) == 0 and len(y) == 0:
        # then this is an empty image
        return None, None
    top_left = y.min(), x.min()
    bottom_right = y.max(), x.max()
    return top_left, bottom_right


def get_total_area_of_and_metadata_of_masks(metadata_df, images_df, rle_encodings_df):
    """
    important note: if there are multiple notes for each image, we use the sum of these masks

    note: the result does not only give the area, but also some metadata for plotting later on
    """
    result = []
    for _, row in metadata_df.iterrows():
        # prepare data for plots
        try:
            uid = row['UID']
            image = get_image_by_uid(images_df, uid)
            mask = get_mask_by_uid(rle_encodings_df, image, uid)
            pixels = np.count_nonzero(mask)

            # save results with additional metadata on the mask and image
            result.append(
                {
                    'TotalArea': pixels,
                    'ViewPosition': row['ViewPosition'],
                    'Sex': row['Sex'],
                    'Age': row['Age'],
                    'Mask': mask,  # get the mask itself for later on when we will create heatmaps
                })
        except Exception as e:
            raise e
            # print(f'could not process image with uid {uid}.\nreason: {e}')
            # continue

    return result