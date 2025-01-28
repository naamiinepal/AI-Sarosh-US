import numpy as np
import pandas as pd
import random

# Set a random seed for reproducibility
random.seed(12345)

# Load the presentation CSV file
df_out_all = pd.read_csv("/mnt/Enterprise2/AI-Sarosh/datasets/famli_csvs/presentation.csv")

# Get unique patient IDs from the dataset
patients = df_out_all.pid.unique()

# Define the percentage splits for validation and test sets
val = 15
test = 15

# Define the output root directory
out_root = "/mnt/Enterprise2/AI-Sarosh/datasets/famli_csvs"

# Shuffle the patient IDs randomly
random.shuffle(patients)

# Calculate the number of patients for the validation and test splits
val_num = int(val * len(patients) / 100.0)
test_num = int(test * len(patients) / 100.0)

# Assign patients to the test, validation, and train sets
test_patients = patients[0:test_num]
val_patients = patients[test_num:test_num + val_num]
train_patients = patients[test_num + val_num:]

# Ensure the split is correct (sum of test, val, and train patients equals total patients)
assert len(test_patients) + len(val_patients) + len(train_patients) == len(patients)

# Save the split datasets (test, val, train) to separate CSV files
df_out_all[df_out_all['pid'].isin(test_patients)].to_csv(f'{out_root}/test/presentation.csv')
df_out_all[df_out_all['pid'].isin(val_patients)].to_csv(f'{out_root}/val/presentation.csv')
df_out_all[df_out_all['pid'].isin(train_patients)].to_csv(f'{out_root}/train/presentation.csv')
