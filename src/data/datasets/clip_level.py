from pathlib import Path
from typing import Callable, Sequence, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImaged, ToTensord, CropImgDropRepeatedChannels

class ClipLevelDataset(Dataset):
    """
    CLIP level Dataset for Ultrasound Video with corresponding labels.
    
    Args:
        image_root_dir: Directory containing ultrasound images
        split_csv_file: CSV file with image split information
        split_label_csv_file: CSV file with label information
        transform: Optional data transformation pipeline
    """
    def __init__(
        self,
        image_root_dir: Union[str, Path],
        split_csv_file: Union[str, Path],
        split_label_csv_file: Union[str, Path],
        transform: Union[Sequence[Callable], Callable, None] = None
    ):
        self.image_root_dir = Path(image_root_dir)
        
        # Read and merge dataframes
        data_df = pd.read_csv(split_csv_file, index_col=0)
        labels_df = pd.read_csv(split_label_csv_file, index_col=0)
        merged_df = pd.merge(
            data_df, labels_df, 
            on=['PatientID', 'StudyInstanceUID', 'fname']
        )[['fname', 'PatientID', 'StudyInstanceUID', 'placenta', 'presentation']]
        
        self.data_arr = merged_df.to_numpy()
        
        # Set default transform if not provided
        self.transform = transform or Compose([
            LoadImaged(reader="PydicomReader", keys=["image"]), 
            ToTensord(keys=["image"]), 
            CropImgDropRepeatedChannels()
        ])

    def __len__(self) -> int:
        return len(self.data_arr)

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> dict:
        """
        Retrieve and transform dataset item.
        
        Returns:
            Transformed dictionary with image and metadata
        """
        data_dict = {
            "image": self.data_arr[idx, 0],
            "fname": self.data_arr[idx, 0],
            "patient": self.data_arr[idx, 1],
            "instance": f"{self.data_arr[idx, 2]}_{str(self.data_arr[idx, 0]).split('/')[-1].split('.')[0]}",
            "placenta": self.data_arr[idx, 3],
            "presentation": self.data_arr[idx, 4],
        }
        
        return self.transform(data_dict)