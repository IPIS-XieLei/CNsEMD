# CNsEMD: An Expert-Annotated Multi-Field-Strength MRI Dataset and a Hyperspherical Manifold Network for Multimodal Cranial Nerve Parcellation

Official implementation of **PHM-Net** and the accompanying **CNsEMD**
dataset for multimodal cranial nerve (CN) Parcellation using
T1-weighted (T1w) and direction-encoded color (DEC) MRI.

# CNsEMD

CNsEMD contains 202 expert-annotated multimodal MRI examinations acquired
at 3T, 5T, and 7T. To our knowledge, upon release it will be the first
public multimodal MRI dataset with expert voxel-level annotations for CN Parcellation.

The public release will include redistributable DEC images derived from the
original diffusion-weighted imaging (DWI) data, expert reference labels, and
fixed training/test split files. The original T1w images will not be
redistributed through CNsEMD. Users should obtain the corresponding T1w images
from the official Human Connectome Project (HCP) and Diff5T repositories.

This repository provides code for data preprocessing, five-fold training,
batch evaluation, field-strength-specific metric summarization, and
single-case inference with pretrained models.

## Release status

The CNsEMD dataset and pretrained model weights have been deposited in
ScienceDB:

**[https://doi.org/10.57760/sciencedb.44096](https://doi.org/10.57760/sciencedb.44096)**

Access is currently restricted. The ScienceDB record and its associated files
will be made publicly available after acceptance of the associated paper.

After release, the ScienceDB deposit will contain:

- `CNsEMD`: redistributable DEC images derived from the original DWI data,
  expert reference labels, and the fixed training/test split files;
- `CNsEMD_weights`: pretrained five-fold checkpoints for PHM-Net, CNTSeg, and
  CNTSeg-v2.

The original T1w images are not included in the ScienceDB deposit. Users should
obtain the corresponding T1w images from the official HCP and Diff5T
repositories and place them in the expected `T1/` directory. For multimodal
training or inference, the T1w images must be correctly paired and spatially
co-registered with the released DEC images.

## Dataset contents

| Subset | Field strength | Matrix size | Voxel size (mm³) | Cases |
| --- | ---: | ---: | ---: | ---: |
| HCP3T | 3T | 145 × 174 × 145 | 1.25 × 1.25 × 1.25 | 102 |
| Diff5T | 5T | 151 × 181 × 151 | 1.20 × 1.20 × 1.20 | 50 |
| HCP7T | 7T | 173 × 207 × 173 | 1.05 × 1.05 × 1.05 | 50 |
| **Total** | **3T/5T/7T** | — | **1.05–1.25** | **202** |

The data and weight directories expected by the provided scripts are organized
as follows. The `T1/` directory is not included in the ScienceDB deposit and
must be populated by users with the corresponding T1w images obtained from the
official HCP and Diff5T repositories.

```text
download_root/
├── CNsEMD/
│   ├── T1/                         # User-provided; not included in the release
│   │   └── <case>-T1.nii.gz
│   ├── DEC/                        # Included in the ScienceDB release
│   │   └── <case>-DEC.nii.gz
│   ├── GT/                         # Included in the ScienceDB release
│   │   └── <case>-label.nii.gz
│   ├── splits_final_train.json     # Included in the ScienceDB release
│   └── splits_final_test.json      # Included in the ScienceDB release
└── CNsEMD_weights/                 # Included in the ScienceDB release
    ├── PHMNet/
    │   ├── fold_0/BEST_MODEL.pth
    │   ├── fold_1/BEST_MODEL.pth
    │   ├── fold_2/BEST_MODEL.pth
    │   ├── fold_3/BEST_MODEL.pth
    │   └── fold_4/BEST_MODEL.pth
    ├── CNTSeg_V1_T1DEC/
    │   ├── fold_0/BEST_MODEL.pth
    │   └── ...
    └── CNTSegV2_Dedicated/
        ├── fold_0/BEST_MODEL.pth
        └── ...
```

File prefixes identify the field strength. For example,
`3_100206-T1.nii.gz`, `3_100206-DEC.nii.gz`, and
`3_100206-label.nii.gz` represent the same 3T case. The DEC image and expert
label are included in the ScienceDB release, whereas the corresponding T1w
image must be obtained separately from the applicable source repository and
placed in the `T1/` directory.

Before preprocessing, users must ensure that each T1w image is correctly
paired and spatially co-registered with its corresponding DEC image. Their
spatial dimensions and affine information must also be consistent.

### Cranial nerve labels

| Label | Anatomical structure |
| ---: | --- |
| 0 | Background |
| 1 | CN II: optic nerve |
| 2 | CN III: oculomotor nerve |
| 3 | CN V: trigeminal nerve |
| 4 | CN VII/VIII: facial and vestibulocochlear nerves |

The annotations represent five bilateral CN pairs. CN VII and CN VIII are
combined into one foreground class because of their close anatomical
proximity.

### Evaluation split

- Training/validation set: 162 cases, including 82 at 3T, 40 at 5T, and 40 at
  7T.
- Independent test set: 40 cases, including 20 at 3T, 10 at 5T, and 10 at 7T.
- The training/validation set is divided into five training and validation
  folds.
- Model selection is performed exclusively using the training and validation
  data.
- The 3T, 5T, and 7T results are subgroup analyses of the same 40-case
  independent test set.

The fixed partitions are stored in `splits_final_train.json` and
`splits_final_test.json`.

# PHM-Net

## Repository structure

```text
CNsEMD_code/
├── CNsEMD_preprocessing_data.py  # Crop volumes and create 2D slices
├── CNsEMD_dataloader.py          # T1w/DEC training dataset
├── CNsEMD_mutilloss.py           # Training losses and utilities
├── CNsEMD_train.py               # Five-fold training and validation
├── CNsEMD_test.py                # Batch ensemble inference and evaluation
├── predict_single_case.py        # Inference for one T1w–DEC subject
├── summarize_357T_metrics.py     # Mixed, 3T, 5T, and 7T summaries
├── utils.py
└── NetModel/
    ├── PHMNet.py
    ├── CNTSeg_V1.py
    └── CNTSegV2_Dedicated.py
```

## Installation

Create a Conda environment:

```bash
conda create -n cnsemd python=3.9
conda activate cnsemd
```

Install PyTorch and torchvision for the CUDA version available on your
machine by following the official
[PyTorch installation guide](https://pytorch.org/get-started/locally/).

Then install the remaining dependencies:

```bash
pip install numpy nibabel SimpleITK scipy scikit-image tqdm thop
```

Run all commands from the repository root so that the local `NetModel` package
and `utils.py` can be imported correctly.

## Quick start: single-case inference

Single-case inference requires a co-registered T1w image and DEC image. The DEC
NIfTI file must have channel-last shape `(X, Y, Z, 3)`, and its spatial shape
and affine information must match those of the T1w image.

Download the pretrained weights from ScienceDB after their public release and
place the weight directory beside `predict_single_case.py`, or specify its
location using `--result-root`.

Run PHM-Net inference using:

```bash
python predict_single_case.py \
  --result-root /path/to/CNsEMD_weights \
  --t1 /path/to/case-T1.nii.gz \
  --dec /path/to/case-DEC.nii.gz
```

Defaults:

- model: `PHMNet`;
- ensemble: folds 0–4;
- device: `cuda:0` when CUDA is available, otherwise CPU;
- output: `<T1-stem>_DEC-<model>-prediction.nii.gz` beside the input T1w image.

All settings except `--t1`, `--dec`, and `--result-root` are optional. For
example:

```bash
python predict_single_case.py \
  --t1 /path/to/case-T1.nii.gz \
  --dec /path/to/case-DEC.nii.gz \
  --model CNTSegV2_Dedicated \
  --result-root /path/to/CNsEMD_weights \
  --output /path/to/prediction.nii.gz \
  --device cuda:1
```

The script center-crops or pads the inputs to `128 × 160 × 128`, applies the
same slice-wise T1w normalization used during evaluation, averages the
selected fold logits, and restores the label map to the original T1w space.

The script does not perform T1w–DEC registration or generate DEC images from
raw DWI data. Users must prepare spatially aligned T1w and DEC images before
running inference.

For `CNTSeg_V1_T1DEC`, the weight directory must additionally contain the two
pretrained modality branches:

```text
CNsEMD_weights/CNTSeg_V1_T1DEC/
├── UnetT1/
│   └── fold_<n>/BEST_MODEL.pth
├── UnetDEC/
│   └── fold_<n>/BEST_MODEL.pth
└── fold_<n>/BEST_MODEL.pth
```

## Data preprocessing

The preprocessing script center-crops each volume to `128 × 160 × 128`, clips
T1w intensities to the 1st–99th percentiles, applies slice-wise z-score
normalization, preserves the DEC values, and saves labeled axial slices for
training. Foreground training slices are augmented by flipping along the first
spatial axis.

> **Warning:** If `--output_root` already exists, the script deletes that
> directory before regenerating it. Do not select a directory containing files
> that must be retained.

```bash
python CNsEMD_preprocessing_data.py \
  --data_root /path/to/download_root/CNsEMD \
  --split_json /path/to/download_root/CNsEMD/splits_final_train.json \
  --test_split_json /path/to/download_root/CNsEMD/splits_final_test.json \
  --output_root /path/to/CNsEMD_preprocessed_data \
  --modality_type two
```

The output structure is:

```text
CNsEMD_preprocessed_data/
├── fold_0/
│   ├── train/
│   │   ├── x_t1_data/
│   │   ├── x_dec_data/
│   │   └── y_label_data/
│   └── val/
│       ├── x_t1_data/
│       ├── x_dec_data/
│       └── y_label_data/
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
└── test_set/
    ├── x_t1_data/
    ├── x_dec_data/
    └── y_label_data/
```

## Training

Supported model names:

- `PHMNet`
- `CNTSegV2_Dedicated`
- `CNTSeg_V1_T1DEC`

Train PHM-Net on all five folds:

```bash
python CNsEMD_train.py \
  --raw_data_root /path/to/download_root/CNsEMD \
  --fornetwork_root /path/to/CNsEMD_preprocessed_data \
  --result_root /path/to/CNsEMD_results \
  --split_json /path/to/download_root/CNsEMD/splits_final_train.json \
  --model PHMNet \
  --gpus 0
```

Important training arguments:

| Argument | Default | Description |
| --- | ---: | --- |
| `--model` | `PHMNet` | Model configuration |
| `--gpus` | `0` | CUDA device index |
| `--batch_size` | `32` | Training batch size |
| `--num_workers` | `4` | Number of DataLoader workers |
| `--epochs` | `200` | Maximum number of epochs |
| `--lr` | `0.002` | Initial learning rate |
| `--folds` | All folds | Comma-separated fold names |

Training uses Adam, `ReduceLROnPlateau`, and early stopping with a patience of
20 epochs. Each fold writes its best checkpoint and training log to:

```text
<result_root>/<model>/fold_<n>/
├── BEST_MODEL.pth
├── training_log.txt
└── Predictions/
```

## Batch ensemble inference and evaluation

`CNsEMD_test.py` loads all five `BEST_MODEL.pth` checkpoints, averages their
logits, reconstructs predictions in the original 3D space, and optionally
evaluates them against the expert labels.

```bash
python CNsEMD_test.py \
  --input_folder /path/to/CNsEMD_preprocessed_data/test_set \
  --gt_folder /path/to/CNsEMD_preprocessed_data/test_set/y_label_data \
  --result_root /path/to/CNsEMD_weights \
  --model PHMNet \
  --mode batch \
  --gpus 0
```

If labels are unavailable, omit `--gt_folder`. Predictions are saved under:

```text
<result_root>/<model>/Ensemble_Results/
├── <case>-ensemble.nii.gz
└── ensemble_metrics_summary.txt
```

The reported metrics are the Dice similarity coefficient, Jaccard index
(Jac), average surface distance (ASD), and average Hausdorff distance (AHD).
Metrics are reported across all foreground classes and separately for each CN
category.

The mixed, 3T, 5T, and 7T summaries can be generated using:

```bash
python summarize_357T_metrics.py \
  --input /path/to/ensemble_metrics_summary.txt
```

## Reproducibility notes

- The `T1/` directory must be populated by users with the corresponding T1w
  images obtained from the official source repositories.
- T1w and DEC volumes must be correctly paired and spatially aligned.
- T1w and DEC volumes must have consistent spatial dimensions and affine
  information.
- DEC volumes must use channel-last layout `(X, Y, Z, 3)`.
- Training and batch evaluation use the fixed five-fold split files.
- Model selection must use only the training and validation folds. The 40-case
  independent test set is reserved for final evaluation.
- Five-fold ensemble inference requires checkpoints for `fold_0` through
  `fold_4`.

## Generating DEC images from DWI data using MRtrix3

The DEC images included in CNsEMD were derived from the original DWI data.
Users may generate compatible DEC images from their own DWI data using
[MRtrix3](https://www.mrtrix.org/). A minimal example is provided below:

```bash
mrconvert /path/to/data.nii.gz /path/to/DWI.mif \
  -fslgrad /path/to/bvecs /path/to/bvals

dwi2tensor /path/to/DWI.mif /path/to/tensor.mif

tensor2metric /path/to/tensor.mif \
  -vector /path/to/DEC.nii.gz
```

The resulting DEC image should have channel-last layout `(X, Y, Z, 3)`.
Before using it for training or inference, ensure that it is spatially aligned
with the corresponding T1w image and that their spatial dimensions and affine
information are consistent.

## Data and code availability

The source code for PHM-Net, model training, evaluation, and single-case
inference is available in this GitHub repository.

The CNsEMD dataset and pretrained model weights have been deposited in
ScienceDB under DOI
[10.57760/sciencedb.44096](https://doi.org/10.57760/sciencedb.44096).
The ScienceDB record is currently access-restricted and will become publicly
available after acceptance of the associated paper.

The public release will include redistributable DEC images derived from the
original DWI data, expert reference labels, fixed training/test split files,
and pretrained model weights. The original T1w images will not be
redistributed through CNsEMD. Users should obtain the corresponding T1w images
from the official HCP and Diff5T repositories and place them in the `T1/`
directory following the naming convention described above.

## Citation

If you use CNsEMD or PHM-Net, please cite the associated paper and dataset.
The final paper citation and BibTeX entry will be added after acceptance.

Dataset identifier:

```text
CNsEMD. ScienceDB. https://doi.org/10.57760/sciencedb.44096
```

## License

The licences for the source code, CNsEMD dataset, and pretrained model weights
will be specified before public release. The released CNsEMD dataset will
include redistributable DWI-derived DEC images, expert reference labels, and
fixed data split files. The original T1w images are not included.

## Contact

For questions about the code or dataset, please open a GitHub issue or contact
[leix@zjut.edu.cn](mailto:leix@zjut.edu.cn).
