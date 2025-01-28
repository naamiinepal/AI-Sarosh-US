from collections.abc import Iterable
from typing import Any
import torch
from torch.nn.functional import pad

def ultrasound_collator_fn(data_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for ultrasound image data with padding.
    
    Args:
        data_samples: List of dictionaries containing image and other data
    
    Returns:
        Batched dictionary with padded images and collected other data
    """
    # Get keys from the first sample
    sample_keys = data_samples[0].keys()
    
    # Initialize output batch dictionary
    batched_data = {}
    
    for key in sample_keys:
        if key == "image":
            # Find the maximum sequence length for padding
            max_sequence_length = max(
                sample['image'].shape[0] for sample in data_samples
            )
            
            # Pad images to consistent length
            batched_data['image'] = [
                pad(
                    sample['image'], 
                    (0, 0, 0, 0, 0, max_sequence_length - sample['image'].shape[0]), 
                    mode="constant", 
                    value=0
                ) 
                for sample in data_samples
            ]
        else:
            # Collect other keys as lists
            batched_data[key] = [
                sample[key] for sample in data_samples
            ]
    
    return batched_data

def famli_collator_fn(data_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for FAMLI dataset with label mapping and padding.
    
    Args:
        data_samples: List of dictionaries containing stacked images and other data
    
    Returns:
        Batched dictionary with padded images, mapped labels, and other data
    """
    # Placenta position label mapping
    #TODO edit this
    PLACENTA_LABEL_MAP = {
        "Cephalic": 0, 
        "Transverse": 1, 
        "Breech": 2
    }
    
    # Get keys from the first sample
    sample_keys = data_samples[0].keys()
    
    # Initialize output batch dictionary
    batched_data = {}
    
    for key in sample_keys:
        if key == "stacked_img":
            # Find the maximum sequence length for padding
            max_sequence_length = max(
                sample['stacked_img'].shape[0] for sample in data_samples
            )
            
            # Pad and stack images
            batched_data['stacked_img'] = torch.stack([
                pad(
                    sample['stacked_img'], 
                    (0, 0, 0, 0, 0, max_sequence_length - sample['stacked_img'].shape[0]), 
                    mode="constant", 
                    value=0
                ) 
                for sample in data_samples
            ])
        else:
            # Collect other keys as lists
            batched_data[key] = [sample[key] for sample in data_samples]
    
    # Map placenta labels to tensor
    #TODO edit this
    batched_data['us_lie'] = torch.stack([
        torch.tensor(PLACENTA_LABEL_MAP[label]) 
        for label in batched_data['us_lie']
    ])
    
    return batched_data