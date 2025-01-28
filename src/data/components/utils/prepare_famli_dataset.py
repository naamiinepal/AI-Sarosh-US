import pydicom as dicom
import matplotlib.pylab as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
import sys
import json
from pathlib import Path
from typing import Any, Optional, Union
import os
import random
random.seed(44)

def main(image_root: Path,
         out_root: Path,
         labels_root: Path,
         val: float,
         test: float):
    """
    Main function to split the DICOM dataset into training, validation, and test sets.
    It reads the DICOM images, extracts metadata, and organizes them into the specified splits.
    It also organizes the associated label files into the corresponding directories.

    Args:
        image_root (Path): The root directory where DICOM image files are stored.
        out_root (Path): The root directory where the output (split datasets) will be saved.
        labels_root (Path): The root directory where the labels CSV files (labels_clip and labels_study) are located.
        val (float): The percentage of data to be used for validation.
        test (float): The percentage of data to be used for testing.

    """
    # Create the necessary output directories
    Path(out_root).mkdir(parents=True, exist_ok=True)
    Path(f'{out_root}/test').mkdir(parents=True, exist_ok=True)
    Path(f'{out_root}/val').mkdir(parents=True, exist_ok=True)
    Path(f'{out_root}/train').mkdir(parents=True, exist_ok=True)

    # Get all DICOM image file paths (both .dcm and .DCM extensions)
    image_paths = glob(f'{image_root}/**/*.dcm', recursive=True) + glob(f'{image_root}/**/*.DCM', recursive=True)

    # Load the labels CSV files
    df_labels_clip = pd.read_csv(f'{labels_root}/labels_clip.csv')
    df_labels_study = pd.read_csv(f'{labels_root}/labels_study.csv')

    # Define the metadata keys we are interested in
    dcm_keys = ['PatientID', 'StudyInstanceUID', 'NumberOfFrames']

    # List to hold the metadata for each image
    meta = []

    # Dictionary for typecasting specific DICOM fields
    typemap = {
        dicom.uid.UID: str,
        dicom.multival.MultiValue: list
    }
    
    def cast(x):
        """Helper function to cast DICOM field values based on their types."""
        return typemap.get(type(x), lambda x: x)(x)

    # Loop through the images and extract metadata
    for i, image_path in enumerate(tqdm(image_paths)):
        with dicom.dcmread(image_path) as obj:
            # Extract metadata values for each DICOM file
            meta.append([image_path] + [cast(obj.get(key, np.nan)) for key in dcm_keys])

    # Create a DataFrame for the metadata
    dfmeta = pd.DataFrame(meta, columns=['fname'] + dcm_keys)

    # Filter out rows where 'NumberOfFrames' is null
    df_out_all = dfmeta[dfmeta['NumberOfFrames'].notnull()][['fname', 'PatientID', 'StudyInstanceUID']]

    # Save the filtered metadata to a CSV file
    df_out_all.to_csv(f'{out_root}/data_all.csv')

    # Get unique patient IDs and shuffle them
    patients = df_out_all.PatientID.unique()
    random.shuffle(patients)

    # Calculate the number of patients for validation and test splits
    val_num = int(val * len(patients) / 100.0)
    test_num = int(test * len(patients) / 100.0)

    # Split the patients into test, validation, and training sets
    test_patients = patients[0:test_num]
    val_patients = patients[test_num:test_num + val_num]
    train_patients = patients[test_num + val_num:]

    # Ensure the split sums correctly
    assert len(test_patients) + len(val_patients) + len(train_patients) == len(patients)

    # Save the metadata for each split (train, val, test) into separate CSVs
    df_out_all[df_out_all['PatientID'].isin(test_patients)].to_csv(f'{out_root}/test/data.csv')
    df_out_all[df_out_all['PatientID'].isin(val_patients)].to_csv(f'{out_root}/val/data.csv')
    df_out_all[df_out_all['PatientID'].isin(train_patients)].to_csv(f'{out_root}/train/data.csv')

    # Save the corresponding labels (clip and study) for each split
    df_labels_clip[df_labels_clip['PatientID'].isin(test_patients)].to_csv(f'{out_root}/test/labels_clip.csv')
    df_labels_clip[df_labels_clip['PatientID'].isin(val_patients)].to_csv(f'{out_root}/val/labels_clip.csv')
    df_labels_clip[df_labels_clip['PatientID'].isin(train_patients)].to_csv(f'{out_root}/train/labels_clip.csv')

    df_labels_study[df_labels_study['PatientID'].isin(test_patients)].to_csv(f'{out_root}/test/labels_study.csv')
    df_labels_study[df_labels_study['PatientID'].isin(val_patients)].to_csv(f'{out_root}/val/labels_study.csv')
    df_labels_study[df_labels_study['PatientID'].isin(train_patients)].to_csv(f'{out_root}/train/labels_study.csv')


if __name__ == "__main__":
    from argparse import ArgumentParser

    # Set up argument parser for command-line arguments
    parser = ArgumentParser()

    # Add arguments for the required paths and percentage splits
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="The root directory of input images (DICOM files)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="The root directory where the output split datasets will be saved",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        required=True,
        help="The root directory where the labels CSV files (labels_clip and labels_study) are located",
    )
    parser.add_argument(
        "--val",
        type=float,
        required=True,
        help="The percentage of the dataset to be used for the validation set",
    )
    parser.add_argument(
        "--test",
        type=float,
        required=True,
        help="The percentage of the dataset to be used for the test set",
    )

    # Parse the arguments and call the main function
    args = parser.parse_args()
    main(**vars(args))
