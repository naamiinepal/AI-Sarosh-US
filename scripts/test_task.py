import os
import logging
import json
# List of datasets
ckpt_path="/mnt/Enterprise2/AI-Sarosh/code-repo/outputs/mvit-all-tag/train/runs/2025-07-02_07-52-34/checkpoints/epoch20-val_f10.835.ckpt"
src_model="mvit"
# Loop over each dataset and run the experiment
TASK='tag'

command = f"python3 src/eval.py \
          +experiment=mvit \
          data=device \
          experiment.name=mvit-trained-on-turbo-quality-{TASK} \
          ckpt_path={ckpt_path} \
          +experiment.task={TASK} \
          +model.net.use_head_only=False \
          +ml_task={TASK} \
          +device=all \
          data.batch_size=16"
          
logging.info(f"RUNNING COMMAND \n{command}")    
      #Run the command
if os.system(command=command) != 0:
    logging.info(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
      