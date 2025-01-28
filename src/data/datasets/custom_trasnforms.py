import torch

class CropImgDropRepeatedChannels:
   """Crop image and drop repeated channels."""
   def __call__(self, sample: dict) -> dict:
       """
       Process image by permuting and selecting first channel.
       
       Args:
           sample: Dictionary containing image tensor
       
       Returns:
           Processed sample with modified image
       """
       image = sample['image']
       image = image.permute(1, 3, 0, 2)  # Reorder dimensions
       image = image[:, 0, :, :]  # Select first channel
       sample['image'] = image
       return sample

        
        