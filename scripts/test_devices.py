import os
import logging
import json
# List of datasets
ckpt_path="/mnt/Enterprise2/AI-Sarosh/code-repo/outputs/mvit-Turbo-us_lie/train/runs/2025-05-25_08-55-56/checkpoints/epoch05-val_f10.802.ckpt"
devices=['VolusonS', 'V830', 'C3HD', 'iQProbe', 'iQ+Probe', 'LOGIQC3Premium']
src_model="mvit"
# Loop over each dataset and run the experiment
TASK='us_lie'

command = f"python3 src/eval_multiple.py \
          +experiment=mvit_multiple_eval \
          data=eval_multiple_devices \
          experiment.name=mvit-trained-on-turbo-best \
          ckpt_path={ckpt_path} \
          +experiment.task={TASK} \
          +model.net.use_head_only=False \
          +model.test_datasets='{json.dumps(devices)}' \
          +ml_task={TASK} \
          data.batch_size=16"
          
logging.info(f"RUNNING COMMAND \n{command}")    
      #Run the command
if os.system(command=command) != 0:
    logging.info(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
      