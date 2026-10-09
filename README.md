# AI-Sarosh-US

Research code for experiments on blind-sweep obstetric ultrasound videos.

Contains training and evaluation pipelines for ultrasound video classification, experiments across different ultrasound devices, and infrastructure for studying domain shift and continual domain adaptation.

The main experiments use an MViT video model for fetal presentation and other ultrasound classification tasks.

## Overview

A typical classification experiment follows:

```text
blind-sweep ultrasound video
        ↓
preprocessed video tensor
        ↓
sample 32 frames
        ↓
resize and normalize
        ↓
MViT
        ↓
video-level prediction
        +
intermediate CLS embedding
```

The extracted embeddings also support representation analysis and continual/domain-adaptation experiments.

## Datasets

The experiments use the FAMLI (Fetal Age Machine Learning Initiative) obstetric ultrasound dataset [1] and AI-Sarosh data.

AI-Sarosh is an internal blind-sweep obstetric ultrasound dataset developed at NAAMII and is planned for public release.

## Tasks

The main clip-level dataset supports the following labels.

### Fetal presentation

`us_lie`

```text
Cephalic      -> 0
Breech        -> 1
Transverse    -> 1
Oblique       -> 1
non-Cephalic  -> 1
```

This gives a binary classification task:

```text
Cephalic vs non-Cephalic
```

### Placenta location

`us_plac`

```text
Anterior  -> 0
Posterior -> 1
```

### Multi-class tag

`tag`

```text
C1 -> 0
C2 -> 1
C3 -> 2
M  -> 3
R1 -> 4
L1 -> 5
```

The repository also contains an MViT configuration for this six-class task.

## Ultrasound device domains

The same training pipeline supports data collected using different ultrasound systems.

Device/domain names appearing in the experiment scripts include:

```text
Turbo
VolusonS
V830
C3HD
iQProbe
iQ+Probe
LOGIQC3Premium
```

There is also support for an `all` dataset containing combined data.

Instead of keeping a separate Hydra config for every device, the repository uses a common device configuration:

```text
configs/data/device.yaml
```

The device name determines which train, validation, and test CSV files are loaded.

The expected structure is:

```text
data/
└── csvs/
    └── quality/
        ├── train/
        │   ├── Turbo.csv
        │   ├── VolusonS.csv
        │   └── ...
        ├── valid/
        │   ├── Turbo.csv
        │   ├── VolusonS.csv
        │   └── ...
        └── test/
            ├── Turbo.csv
            ├── VolusonS.csv
            └── ...
```

For example:

```bash
+device=Turbo
```

selects:

```text
data/csvs/quality/train/Turbo.csv
data/csvs/quality/valid/Turbo.csv
data/csvs/quality/test/Turbo.csv
```

This setup makes it straightforward to train on one device and evaluate on another.

## Data loading

The main clip-level dataset is implemented in:

```text
src/data/datasets/clip_level.py
```

Each CSV contains information such as:

```text
study_id
file_path_old
numberofframes
task label
```

Ultrasound videos are loaded from preprocessed PyTorch tensors.

For each sample, the dataset:

1. loads the ultrasound video
2. uniformly samples 32 frames
3. converts grayscale frames to three channels
4. optionally adjusts spatial dimensions using DICOM pixel spacing
5. resizes the frames to `224 x 224`
6. applies ImageNet normalization

The resulting batch has the form:

```text
B x T x C x H x W
```

The training dataloader also supports weighted sampling to reduce the effect of class imbalance.

## MViT model

The main video model is implemented in:

```text
src/models/components/mvit.py
```

It uses the pretrained:

```text
mvit_base_32x3
```

model from PyTorchVideo.

The original classification projection is replaced with a task-specific classification head.

The model also extracts a CLS representation from an intermediate transformer block:

```text
video
  ↓
MViT backbone
  ↓
CLS representation
  ↓
classification head
```

By default, the representation comes from block 15.

The forward pass provides both:

```text
prediction logits
CLS embedding
```

This allows the same network to act as both a video classifier and a feature extractor.

## Training

Training uses PyTorch Lightning and Hydra.

The main fetal-presentation experiment config is:

```text
configs/experiment/mvit_ind_lie.yaml
```

An example run for the Turbo device is:

```bash
python src/train.py \
    experiment=mvit_ind_lie \
    +device=Turbo \
    +ml_task=us_lie
```

The default MViT configuration uses settings such as:

```text
input size:       224 x 224
frames:           32
batch size:       8
optimizer:        Adam
learning rate:    3e-5
precision:        16-mixed
classes:          2
```

Validation F1 is used for checkpoint selection.

Changing the device only requires changing the Hydra argument:

```bash
python src/train.py \
    experiment=mvit_ind_lie \
    +device=V830 \
    +ml_task=us_lie
```

or:

```bash
python src/train.py \
    experiment=mvit_ind_lie \
    +device=iQProbe \
    +ml_task=us_lie
```

## Six-class experiment

The repository also contains:

```text
configs/experiment/mvit_ind_plac.yaml
```

The current configuration uses:

```text
num_classes: 6
ml_task: tag
```

and can be run as:

```bash
python src/train.py \
    experiment=mvit_ind_plac \
    +device=Turbo \
    +ml_task=tag
```

Despite the `plac` name in the config filename, the current configuration connects to the six-class `tag` label rather than the binary `us_plac` label.

## Predictions and embeddings

After model training, the pipeline can run prediction and save both outputs and intermediate representations.

The prediction file is:

```text
predictions/preds_and_embed.csv
```

and contains information such as:

```text
study_id
file_path
logit
output
gt
embed
```

The `embed` field contains the intermediate MViT CLS representation.

These representations are useful for inspecting domain differences and for feature-space adaptation experiments.

## Cross-device evaluation

Focus on generalization across ultrasound devices.

For example:

```text
train on Turbo
       ↓
evaluate on VolusonS
evaluate on V830
evaluate on C3HD
evaluate on iQProbe
evaluate on iQ+Probe
evaluate on LOGIQC3Premium
```

The main code for this is in:

```text
src/eval_multiple.py
scripts/test_devices.py
```

`eval_multiple.py` constructs test datasets for multiple target devices and evaluates the same checkpoint across them.

This setup makes it possible to measure how performance changes when the acquisition device and data distribution change.

## Continual domain adaptation

The repository also contains infrastructure for continual unsupervised domain adaptation across ultrasound device domains.

The setting considers a model that starts with a labeled source domain and then encounters new unlabeled target domains sequentially:

```text
labeled source device
        ↓
train MViT
        ↓
extract latent representations
        ↓
model class-conditional feature distribution
        ↓
unlabeled target device D2
        ↓
adapt model
        ↓
update representation distribution / memory
        ↓
unlabeled target device D3
        ↓
adapt again
        ↓
...
```

The goal is to adapt the model as new device domains arrive without retraining from scratch and while reducing forgetting of previously encountered domains.

The feature-space adaptation setup is inspired by:

> Mohammad Rostami.  
> **Lifelong Domain Adaptation via Consolidated Internal Distribution.**  
> Advances in Neural Information Processing Systems (NeurIPS), 2021.

The approach models the learned internal feature distribution using a Gaussian Mixture Model (GMM). Representative samples can be maintained in a memory buffer, while pseudo-representations sampled from the internal distribution provide information about previously learned domains during adaptation.

The MViT implementation contains a `use_head_only` option that allows a feature representation to bypass the video backbone and pass directly through the classifier:

```text
GMM / latent representation
          ↓
   classification head
          ↓
       prediction
```

Together with the CLS embedding extraction, this provides the main model-side support for feature-space continual adaptation.

The scripts:

```text
scripts/train_indv.py
scripts/train_indv_plac.py
```

Expects `gmm.py` and `buffer.py` utilities, based on the paper by Rostami et. al. for the complete continual adaptation experiment.

## Study-level prediction

A single patient or study can contain several blind ultrasound sweeps.

The repository includes experiments for combining predictions from several sweeps into a single study-level decision:

```text
scripts/evaltasks.py
```

Aggregation strategies include:

```text
uniform averaging
confidence-weighted averaging
entropy-weighted averaging
maximum-confidence prediction
top-3 confidence averaging
logit-magnitude weighted averaging
```

This allows information from multiple sweeps to contribute to the final prediction instead of treating every sweep independently.

## FAMLI study-level pipeline

The repository also contains a study-level FAMLI pipeline:

```text
configs/data/famli.yaml
src/data/datasets/famli_study_level.py
src/data/datasets/patient_level.py
```

This setup groups multiple sweeps belonging to the same study and works with stacked or preprocessed ultrasound data.

Some paths in `famli.yaml` point to the original data environment and need to be changed for a different machine.

## Other models

The repository contains additional video-modeling baselines alongside MViT.

### Simple baseline

```text
src/models/components/baseline_net.py
configs/experiment/baseline.yaml
```

This model extracts visual features from individual frames and combines the frame representations across time for video classification.

### Blind-sweep fetal presentation baseline

```text
src/models/components/google_baselines.py
configs/experiment/google_fp.yaml
```

This implementation contains a MobileNet-based visual encoder followed by recurrent temporal modeling for fetal-presentation prediction.

The same file also includes a gestational-age regression model.

These models provide alternative baselines to the MViT experiments.

## Dataset preparation

Dataset preparation utilities are available under:

```text
src/data/components/utils/
```

including:

```text
prepare_famli_dataset.py
split_famli_data.py
```

The utilities extract metadata from ultrasound studies and create train, validation, and test splits.

The FAMLI splitting code operates at the patient level so that sweeps belonging to the same patient stay within the same split.

## Repository structure

```text
configs/
    data/
        device.yaml
        famli.yaml

    experiment/
        mvit_ind_lie.yaml
        mvit_ind_plac.yaml
        baseline.yaml
        google_fp.yaml

    model/
        mvit.yaml
        baseline.yaml
        google_fetal_presentation.yaml

src/
    data/
        datasets/
            clip_level.py
            famli_study_level.py
            patient_level.py

        components/
            utils/

    models/
        components/
            mvit.py
            baseline_net.py
            google_baselines.py

        famli_module.py

    train.py
    eval.py
    eval_multiple.py

scripts/
    train_indv.py
    train_indv_plac.py
    test_devices.py
    evaltasks.py
    prepare_dataset.sh

notebooks/
tests/
```

## Setup

Install the dependencies with:

```bash
pip install -r requirements.txt
```

or create the provided Conda environment:

```bash
conda env create -f environment.yaml
```

The main dependencies include:

```text
PyTorch
PyTorch Lightning
PyTorchVideo
MONAI
Hydra
TorchMetrics
Weights & Biases
pandas
scikit-learn
```

## Notes

The ultrasound datasets are not included in the repository.

Some scripts and configuration files contain paths from the original compute environment and need to be changed for another setup.

The repository includes ultrasound video classification, representation extraction, device-specific training, cross-device evaluation, study-level aggregation, and infrastructure for continual domain adaptation.

## References

[1] T. Pokaprakarn et al.  
**AI Estimation of Gestational Age from Blind Ultrasound Sweeps in Low-Resource Settings.**  
NEJM Evidence, 1(5), 2022. doi:10.1056/EVIDoa2100058.

The continual domain-adaptation experiments build on ideas from:

```bibtex
@inproceedings{rostami2021lifelong,
  title={Lifelong Domain Adaptation via Consolidated Internal Distribution},
  author={Rostami, Mohammad},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  pages={11172--11183},
  year={2021}
}
```
