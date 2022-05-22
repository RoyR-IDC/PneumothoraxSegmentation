import os
import sys
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

# project imports
from HW2.PneumothoraxSegmentationProject import FAST_LOAD_MODE, WANTED_IMAGE_SIZE, SELECTIVE_LOAD_MODE, SUBSET_IDS_PATH, \
    DEBUG_MODE, FAST_LOAD_NUM_OF_SAMPLES
from HW2.PneumothoraxSegmentationProject.Utilities.MaskUtilities import get_mask_from_rle_encodings


class SIIMDataset(Dataset):
    def __init__(self, dcm_files_path: str, rle_encodings_filepath: str):
        print(f'entered __init__ of SIIMDataset')
        self._dcm_files_path = dcm_files_path
        self._rle_encodings_filepath = rle_encodings_filepath

        # Get IDs to filter by if requested
        if SELECTIVE_LOAD_MODE is True:
            temp = pd.read_csv(SUBSET_IDS_PATH, delimiter=",")
            self._subset_ids = temp['ImageId'].to_list()

        # Generate the patients dataframe
        self._dataframe = self._get_dataframe()
        print(f'finished __init__ of SIIMDataset')

    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self, index):
        if len(self._dataframe) == 0:
            raise AssertionError('Dataset is empty - therefore __getitem__ fails')
        curr_item = self._dataframe.iloc[index]
        curr_item = curr_item.to_dict()
        return curr_item

    def _get_dcm_filenames_from_dcm_files_path(self, path: str) -> list:
        print(f'entered `_get_dcm_filenames_from_dcm_files_path`')
        print(f'    path given was: {path} verifying path exists: {os.path.isdir(path)}')

        # TODO: improve implementation of this function

        temp = [os.path.join(path, item) for item in os.listdir(path) if (not item.startswith('.'))]

        temp2 = []
        for item in temp:
            t = os.listdir(item)
            if len(t) < 1:
                continue
            new_path = os.path.join(item, t[-1])
            temp2.append(new_path)

        file_names = []

        for item in temp2:
            t = os.listdir(item)
            if len(t) < 1:
                continue
            new_path = os.path.join(item, t[-1])
            if '.dcm' not in new_path:
                continue
            # get subset
            dcm_file_name = t[-1].replace('.dcm', '')
            if SELECTIVE_LOAD_MODE is True:
                if not any(dcm_file_name in item for item in self._subset_ids):
                    continue
            file_names.append({'dcm_file_name': dcm_file_name, 'full_file_name': new_path})

        self._file_names = file_names

        print(f'finished `_get_dcm_filenames_from_dcm_files_path`')
        return file_names

    def _get_rle_encodings_df(self, filepath) -> pd.DataFrame:
        print(f'entered `_get_rle_encodings_df`')
        # read train-rle.csv
        rle_encodings_df = pd.read_csv(filepath, delimiter=",")
        rle_encodings_df.rename(columns={" EncodedPixels": "EncodedPixels", "ImageId": "UID"}, inplace=True)
        print(f'finished `_get_rle_encodings_df`')
        return rle_encodings_df

    def _extract_data_from_dicom_format_to_dict(self, data) -> dict:
        # create a new empty patient
        patient = dict()

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
        patient["OriginalImage"] = data.pixel_array

        return patient

    def _extract_features_from_encodings(self, patient: dict, rle_encodings_df: pd.DataFrame) -> dict:
        try:
            matching_records = rle_encodings_df[rle_encodings_df["UID"] == patient["UID"]]
            rle_encodings = matching_records['EncodedPixels'].to_list()
            patient["Label"] = 'Healthy' if rle_encodings == ['-1'] else 'Pneumothorax'
            patient["OriginalMask"] = get_mask_from_rle_encodings(
                rle_encodings=rle_encodings,
                img_width=patient["OriginalImage"].shape[1],
                img_height=patient["OriginalImage"].shape[0])
            patient["NumOfEncodings"] = 0 if rle_encodings == ['-1'] else len(rle_encodings)

        except:
            patient["Label"] = 'NoLabel'
            patient["OriginalMask"] = np.zeros([WANTED_IMAGE_SIZE, WANTED_IMAGE_SIZE])
            patient["NumOfEncodings"] = 0

        return patient

    def _perform_image_and_mask_preprocess(self, patient):
        try:
            transform = T.Compose([
                T.Resize(WANTED_IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ])

            # convert images to PIL format so that they can be transformed using torchbvision
            patient["OriginalImage"] = Image.fromarray(patient["OriginalImage"])
            patient["OriginalMask"] = Image.fromarray(patient["OriginalMask"])

            # transform
            patient["Image"] = transform(patient["OriginalImage"])
            patient["Mask"] = transform(patient["OriginalMask"])

        except Exception as e:
            print(f'problem with image resize ! ', patient['UID'], f'problem was {type(e), e}')
            patient["OriginalImage"] = np.zeros([WANTED_IMAGE_SIZE, WANTED_IMAGE_SIZE])
            patient["Mask"] = np.zeros([WANTED_IMAGE_SIZE, WANTED_IMAGE_SIZE])
            raise e  # could not process

        return patient

    def _get_dataframe(self) -> pd.DataFrame:
        print(f'entered `_get_dataframe`')

        # Preparations
        patients = pd.DataFrame()
        file_names = self._get_dcm_filenames_from_dcm_files_path(path=self._dcm_files_path)
        rle_encodings_df = self._get_rle_encodings_df(filepath=self._rle_encodings_filepath)

        # Iterate
        for index, names in enumerate(file_names):
            # unpack
            dcm_file_name, full_file_name = names['dcm_file_name'], names['full_file_name']

            # create a smaller dataset if requested
            if FAST_LOAD_MODE is True and index > FAST_LOAD_NUM_OF_SAMPLES:
                break

            # load patient data from DICOM format
            try:
                data = pydicom.dcmread(full_file_name)
            except Exception as e:
                print(f'{type(e)} : could not read {full_file_name}. problem was {e}')
                continue

            # create patient record
            patient = self._extract_data_from_dicom_format_to_dict(data=data)
            patient = self._extract_features_from_encodings(patient=patient, rle_encodings_df=rle_encodings_df)
            patient = self._perform_image_and_mask_preprocess(patient=patient)

            # cleanup #1
            del data
            patient.pop("OriginalImage", None)  # not needed anymore
            patient.pop("OriginalMask", None)  # not needed anymore

            # finally - append the record to the dataframe
            patients = patients.append(patient, ignore_index=True)

            # debug msg
            if DEBUG_MODE is True:
                if len(patients) % 400 == 0:
                    print(f'len(patients) {len(patients)}')

        # return the dataframe as output
        print(f'finished `_get_dataframe`')
        return patients

    def export_to_csv(self, path):
        np.set_printoptions(threshold=sys.maxsize)
        self._dataframe.to_csv(path)

    def get_patients_dataframe(self):
        return self._dataframe


