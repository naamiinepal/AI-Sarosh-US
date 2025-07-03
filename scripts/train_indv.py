import os
from get_best import get_latest_run_dir, get_best_checkpoint
import pandas as pd
import numpy as np
import json
import gmm
import torch
from buffer import get_buffer
import shutil
from datetime import datetime
#model is model.yaml file
#dataset is dataset.yaml file
import logging
TASK='us_lie'

import time
import subprocess

# Sleep for 6 hours (in seconds)


# Configure logging
logging.basicConfig(
    filename='/mnt/Enterprise2/AI-Sarosh/code-repo/logs/cuda_training/train_quality.log',            # log file name
    level=logging.DEBUG,                 # log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # log format
)

root_op_dir = '/mnt/Enterprise3/kanchan/us_uda/outputs'


def train_indv_devices(devices):
    for src_ds in devices:
        exp_name = f'mvit-{src_ds}-{TASK}'
        command = f"python3 src/train.py \
                        experiment=mvit_ind \
                        data=device.yaml \
                        experiment.name={exp_name} \
                        +device={src_ds} \
                        +model.net.use_head_only=False \
                        +ml_task={TASK}"


        print(f"RUNNING COMMAND \n{command}")

            # Run the command
        if os.system(command=command) != 0:
            print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
            exit()
            
devs = {'all':1}

train_indv_devices(devs.keys())
        
    