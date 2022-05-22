import cv2
import numpy as np
import pandas as pd

from HW2.PneumothoraxSegmentationProject.Utilities.MaskUtilities import rle2mask


def generate_metadata_dataframe(train_metadata: list, masks: pd.DataFrame):
    patients = pd.DataFrame()
    # todo remove unwanted feature
    for data in train_metadata:
        patient = dict()
        # patient = data.__dict__
        # save the wanted features from the dicom foramt
        patient["UID"] = data.SOPInstanceUID
        patient["PatientID"] = data.PatientID
        patient["Age"] = data.PatientAge
        patient["Sex"] = data.PatientSex
        patient["Modality"] = data.Modality
        patient["BodyPart"] = data.BodyPartExamined
        patient["ViewPosition"] = data.ViewPosition
        patient["Columns"] = data.Columns
        patient["Rows"] = data.Rows
        patient["PatientOrientation"] = data.PatientOrientation
        patient["PhotometricInterpretation"] = data.PhotometricInterpretation
        patient["PixelSpacing"] = data.PixelSpacing
        patient["SamplesPerPixel"] = data.SamplesPerPixel
        patient["PixelSpacing"] = data.PixelSpacing

        # add a label to the data - if the patient has the disease or not
        try:
            encoded_pixels = masks[masks["ImageId"] == patient["UID"]].values[0][1]
            # patient["EncodedPixels"] = encoded_pixels
            patient["Label"] = 'Healthy' if encoded_pixels == '-1' else 'Pneumothorax'
        except:
            patient["Label"] = 'NoLabel'

        patients = patients.append(patient, ignore_index=True)

    # return the dataframe as output
    return patients


def get_image_by_uid(images_df: pd.DataFrame, uid):
    image = images_df[
        images_df.UID == uid].Image.item()  # item is added because the result is a series object with 1 element
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def get_mask_by_uid(rle_encodings_df: pd.DataFrame, image, uid):
    # create the mask
    # note: there can be more than one RLE encoding per image
    rle_encodings = rle_encodings_df[rle_encodings_df.ImageId == uid].EncodedPixels
    final_mask = None
    for rle_encoding in rle_encodings.to_list():
        current_mask = rle2mask(rle=rle_encoding, width=image.shape[0], height=image.shape[1])
        if final_mask is None:
            final_mask = current_mask
        else:
            # print(f'another mask is added')
            final_mask += current_mask  # Important logic

    # mask needs to be rotated to fit the original image
    mask = final_mask
    mask = np.rot90(mask, 3)  # rotating three times 90 to the right place
    mask = np.flip(mask, axis=1)

    return mask