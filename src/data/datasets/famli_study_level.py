from pathlib import Path
from typing import Any, Callable, Sequence, Union
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, Resized, ToTensord
from torch.utils.data import Dataset

class FamliStudyLevelDataset(Dataset):
    """
    Dataset for Ultrasound Study-Level Data.
    
    Args:
        image_root_dir: Root directory for images
        split_csv_file: CSV file with study information
        transform: Optional data transformation pipeline
        task: Specific task label (default: "us_lie")
    """
    def __init__(
        self,
        image_root_dir: Union[str, Path],
        split_csv_file: Union[str, Path],
        transform: Union[Sequence[Callable], Callable, None] = None,
        task: str = "us_lie",
        sampling_interval: int = 4
    ):
        self.image_root_dir = Path(image_root_dir)
        self.task = task
        self.sampling_interval = sampling_interval
        # Process dataframe
        merged_df = pd.read_csv(split_csv_file)
        data_arr = merged_df.groupby(['study_id', 'pid', task], as_index=False)[['file_path']]
        
        self.data_arr = [
            {
                "meta": np.array(p_key), 
                "files": data.to_numpy()
            } 
            for (p_key, data) in list(data_arr)
        ]
        
        # Set default transform if not provided
        self.transform = transform or Compose([
            ToTensord(keys=["stacked_img"]), 
            Resized(keys=["stacked_img"], spatial_size=450, size_mode="longest")
        ])

    def __len__(self) -> int:
        return len(self.data_arr)

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> dict:
        """
        Retrieve and transform dataset item.
        
        Returns:
            Transformed dictionary with study data and images
        """
        data_dict = {
            "fnames": self.data_arr[idx]['files'],
            "study": self.data_arr[idx]['meta'][0],
            "patient": self.data_arr[idx]['meta'][1],
            f"{self.task}": self.data_arr[idx]['meta'][2],
        }
        
        # Load stacked images with subsampling
        stacked_img = np.load(f'/mnt/Enterprise2/AI-Sarosh/datasets/stacked_famli/{data_dict["study"]}.npy')
        data_dict["stacked_img"] = stacked_img[::self.sampling_interval, :, :]
        res = self.transform(data_dict)
        res["stacked_img"] = torch.unsqueeze(res["stacked_img"], dim=1)
        return res