from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.datasets.collater import famli_collator_fn
import torch
from torch.utils.data import WeightedRandomSampler



class BaseDataModule(LightningDataModule):
   """
   Flexible Lightning DataModule for handling multiple datasets with configurable splits.

   Manages dataset preparation, splitting, and DataLoader creation for machine learning workflows.

   Args:
       train_dataset: Training dataset(s)
       val_dataset: Validation dataset(s), optional
       test_dataset: Test dataset(s), optional
       pred_dataset: Prediction dataset(s), optional
       train_val_split: Proportion/size of train-validation split
       batch_size: Number of samples per batch
       num_workers: Number of subprocess workers for data loading
       pin_memory: Use pinned memory for faster data transfer
       collate_fn: Custom function for batching data
   """
   def __init__(
       self,
       train_dataset: Dataset | List[Dataset],
       val_dataset: Optional[Dataset | List[Dataset]] = None,
       test_dataset: Optional[Dataset | List[Dataset]] = None,
       pred_dataset: Optional[Dataset | List[Dataset]] = None,
       train_val_split: Tuple[float, float] | Tuple[int, int] = (0.8, 0.2),
       batch_size: int = 4,
       num_workers: int = 2,
       pin_memory: bool = False,
       collate_fn: Callable | None = famli_collator_fn
   ):
       super().__init__()
       self.save_hyperparameters(
           logger=False,
           ignore=["train_dataset", "val_dataset", "test_dataset", "pred_dataset"],
       )
       self.batch_size = batch_size
       self.num_workers = num_workers
       self.pin_memory = pin_memory
        
       # Process datasets with flexible handling
       self.data_train = ConcatDataset(train_dataset) if isinstance(train_dataset, Iterable) else train_dataset
       
       # Validation dataset handling
       if val_dataset is None:
           self.data_train, self.data_val = random_split(
               dataset=self.data_train, lengths=train_val_split
           )
       else:
           self.data_val = ConcatDataset(val_dataset) if isinstance(val_dataset, Iterable) else val_dataset

       # Test dataset handling
       if test_dataset is None:
           self.data_val, self.data_test = random_split(
               dataset=self.data_val, lengths=[0.5, 0.5]
           )
       else:
           self.data_test = ConcatDataset(test_dataset) if isinstance(test_dataset, Iterable) else test_dataset

       # Prediction dataset handling
       self.data_pred = (
           ConcatDataset(pred_dataset) if isinstance(pred_dataset, Iterable) else 
           pred_dataset if pred_dataset is not None else 
           self.data_train
       )
       self.collate_fn = collate_fn

   def _create_dataloader(self, dataset, shuffle=False):
       """
       Create standard DataLoader with configured parameters.

       Args:
           dataset: Dataset to load
           shuffle: Whether to shuffle data

       Returns:
           Configured DataLoader
       """
       return DataLoader(
           dataset=dataset,
           batch_size=self.batch_size,
           num_workers=self.num_workers,
           pin_memory=self.pin_memory,
           shuffle=shuffle,
       )

   def train_dataloader(self):
       """Create DataLoader for training data."""
       return self._create_dataloader(self.data_train, shuffle=True)

   def val_dataloader(self):
       """Create DataLoader for validation data."""
       return self._create_dataloader(self.data_val, shuffle=False)

   def test_dataloader(self):
       """Create DataLoader for test data."""
       return self._create_dataloader(self.data_test, shuffle=False)

   def predict_dataloader(self):
       """Create DataLoader for prediction data."""
       return self._create_dataloader(self.data_pred, shuffle=False)

   def teardown(self, stage: Optional[str] = None):
       """Clean up after training or testing."""
       pass

   def state_dict(self):
       """Retrieve additional state for checkpointing."""
       return {}

   def load_state_dict(self, state_dict: Dict[str, Any]):
       """Restore state from checkpoint."""
       pass


class BalancedDataModule(BaseDataModule):
     # Check if the dataset has a method to get weights
      
    def train_dataloader(self):
        sample_weights = self.data_train.get_weights()
        
        if not hasattr(self.data_train, "get_weights"):
            raise AttributeError("Dataset must implement a `get_weights()` method for balancing.")
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=sampler,
        )
        
if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)