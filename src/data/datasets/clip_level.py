from pathlib import Path
from typing import Callable, Sequence, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose,ToTensord, Resized, NormalizeIntensityd
import re
import numpy as np
from collections import Counter




LABEL_ENCODER = {"us_lie_label":{"Cephalic":0, "Breech":1, "Transverse":1, "Oblique":1, "non-Cephalic":1, 1:1, 0:0, 2:1, 3:1, 4:1},
"us_lie_x":{"Cephalic":0, "Breech":1, "Transverse":1, "Oblique":1},
"us_lie":{"Cephalic":0, "Breech":1, "Transverse":1, "Oblique":1, "non-Cephalic":1, 1:1, 0:0, 2:1, 3:1, 4:1},
"us_plac":{"Anterior":0, "Posterior":1},
"tag":{"C1":0, "C2":1,"C3":2, "M":3, "R1":4, "L1":5}}

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
        transform: Union[Sequence[Callable], Callable, None] = None,
        sampling_interval: Union[int,str] = "uniform",
        uniform_samples_num:int = 32,
        task: str = "us_lie",
        spatial_size: int = 224,
    ):
       
        
        self.image_root_dir = Path(image_root_dir)
        self.task = task
        
        if (type(sampling_interval)== str) and (sampling_interval == "uniform"):
            
            self.sampling_interval = None
            self.uniform_samples_num = uniform_samples_num
        else:
            self.sampling_interval = sampling_interval
        # Read and merge dataframes
        data_df = pd.read_csv(split_csv_file).sample(frac=1, random_state=1234).reset_index(drop=True)
        data_df = data_df[data_df["numberofframes"] >1 ].reset_index(drop=False)[['study_id', 'file_path_old', f'{task}', 'numberofframes']]
        # data_df = pd.read_csv(split_csv_file)[['study_id_x', 'file_path', f'{task}']].sample(frac=1, random_state=1234).reset_index(drop=True)
        #file_path, #{task}_label

        # data_df = data_df[data_df[f'{task}'].str.strip() != 'Variable / NA']
        # data_df[data_df[f'{task}'] != 'Variable / NA']



        data_df.dropna(inplace=True)
        self.data_arr = data_df.to_numpy()
        # Set default transform if not provided
        self.transform = transform or Compose([
            ToTensord(keys=["img", 'gt']), 
            Resized(keys=["img"], spatial_size=(None, spatial_size,spatial_size)),
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] [0.1017, 0.1614]
            NormalizeIntensityd(keys=["img"], subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225], channel_wise=True)
        ])
        self.labels = data_df[f'{self.task}'].map(LABEL_ENCODER[f'{self.task}']).tolist()


    def __len__(self) -> int:
        return len(self.data_arr)
    
    def get_weights(self):
        
        class_counts = Counter(self.labels)
        num_samples = len(self.labels)

        #Compute weight for each sample
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in self.labels]
        return sample_weights
        

    def __getitem__(self, idx: Union[int, slice, Sequence[int]]) -> dict:
        """
        Retrieve and transform dataset item.
        
        Returns:
            Transformed dictionary with image and metadata
        """
        mp4= False
        fname = self.data_arr[idx, 1]
        if fname.endswith(".pth") and not fname.endswith("__32.pth"):
            fname = fname.replace(".pth", "__32.pth")
        if fname.endswith(".mp4"):
            fname = fname.replace(".mp4", "__32.pth")
            mp4= True

       
        data_dict = {
            "study_id_x": self.data_arr[idx, 0],
            "patient_id": re.match(r"^(.*)-[^-]+$",self.data_arr[idx, 0]).group(1),
            "fname": fname,
            "gt": torch.tensor(LABEL_ENCODER[f"{self.task}"][self.data_arr[idx, 2]]),
}
        data_dict["img"] =torch.load(data_dict["fname"], weights_only = True)["data"]
        


        if not mp4:
            pixel_spacing = torch.load(data_dict["fname"], weights_only = True)["PixelSpacing"]

        if self.sampling_interval is None:
            
            self.sampling_interval = int(data_dict['img'].shape[2]/ self.uniform_samples_num)
           
        data_dict["img"] = data_dict["img"][ :, :, ::self.sampling_interval][:, :, :self.uniform_samples_num]
        data_dict["img"] = data_dict["img"].unsqueeze(dim=0).expand(3, -1, -1, -1).permute(0,3,1,2)
        if not mp4:
            data_dict["img"] = torch.nn.functional.interpolate(data_dict["img"], (int(data_dict["img"].shape[2]*pixel_spacing[0]/0.75), int(data_dict["img"].shape[3]*pixel_spacing[1]/0.75)))
        data_dict = self.transform(data_dict)
        data_dict["img"] = data_dict["img"].permute(1,0,2,3)
        return data_dict

