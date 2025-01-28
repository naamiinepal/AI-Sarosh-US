from pathlib import Path

import pytest
import torch
import sys

sys.path.append("/mnt/Enterprise2/AI-Sarosh/code-repo")
from src.data.base_datamodule import BaseDataModule
from src.data.mnist_datamodule import MNISTDataModule
from src.data.datasets.clip_level import ClipLevelDataset
from src.data.datasets.famli_study_level import FamliStudyLevelDataset
from src.data.datasets.collater import famli_collator_fn

@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

@pytest.mark.parametrize("batch_size", [32, 128])
def test_us_datamodule(batch_size: int) -> None:

    
    ds = ClipLevelDataset(image_root_dir="/mnt/Enterprise2/AI-Sarosh/datasets/samples",split_csv_file="/mnt/Enterprise2/AI-Sarosh/datasets/csvs/train/data.csv",split_label_csv_file="/mnt/Enterprise2/AI-Sarosh/datasets/csvs/train/labels_clip.csv")

    dm = BaseDataModule(train_dataset=ds, val_dataset=ds, test_dataset=ds, batch_size=batch_size, collate_fn=us_collator_fn)




    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 241*3

    batch = next(iter(dm.train_dataloader()))
    x= batch

    
    
# test_us_datamodule(batch_size=4)
@pytest.mark.parametrize("batch_size", [32, 128])
def test_famli_us_datamodule(batch_size: int) -> None:

    
    ds = FamliStudyLevelDataset(image_root_dir="/home/kanchan",split_csv_file="/mnt/Enterprise2/AI-Sarosh/datasets/famli_csvs/merged-all.csv")

    dm = BaseDataModule(train_dataset=ds, val_dataset=ds, test_dataset=ds, batch_size=batch_size, collate_fn=famli_collator_fn)




    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    # assert num_datapoints == 241*3

    batch = next(iter(dm.train_dataloader()))
    x= batch
    
    
test_famli_us_datamodule(batch_size=4)