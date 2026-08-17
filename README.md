# CNsEMD: An Expert-Annotated Multi-Field-Strength MRI Dataset and a Hyperspherical Manifold Network for Multimodal Cranial Nerve Tract Segmentation

Official implementation of **PHM-Net** and the accompanying **CNsEMD**
dataset for multimodal cranial nerve (CN) tract segmentation from T1-weighted
(T1w) and direction-encoded color (DEC) MRI.

# CNsEMD

CNsEMD contains 202 expert-annotated multimodal MRI examinations acquired
at 3T, 5T, and 7T. To our knowledge, upon release it will be the first
public multimodal MRI dataset with expert voxel-level annotations for CN tract segmentation.
This repository provides code for data preprocessing, five-fold training,
batch evaluation, field-strength-specific metric summarization, and
single-case inference with pretrained models.

## Release status

The CNsEMD dataset and pretrained weights have been deposited in ScienceDB:

**[https://doi.org/10.57760/sciencedb.44096](https://doi.org/10.57760/sciencedb.44096)**

Access is currently restricted; the data will be made publicly available upon acceptance of the associated paper.

After release, the ScienceDB deposit will contain:

- `CNsEMD`: T1w images (The T1w images are not included in CNsEMD. Users can obtain the corresponding T1w images from the official Human Connectome Project (HCP) and Diff5T repositories.), DEC images, expert reference labels, and the fixed training/test split files;
- `CNsEMD_weights`: pretrained five-fold checkpoints for PHM-Net and the two
  comparison models (CNTSeg and CNTSeg-v2) used by the inference scripts.

## Dataset contents

| Subset | Field strength | Matrix size | Voxel size (mm³) | Cases |
| --- | ---: | ---: | ---: | ---: |
| HCP3T | 3T | 145 × 174 × 145 | 1.25 × 1.25 × 1.25 | 102 |
| Diff5T | 5T | 151 × 181 × 151 | 1.20 × 1.20 × 1.20 | 50 |
| HCP7T | 7T | 173 × 207 × 173 | 1.05 × 1.05 × 1.05 | 50 |
| **Total** | **3T/5T/7T** | — | **1.05–1.25** | **202** |

The released data are organized as follows:

```text
download_root/
├── CNsEMD/
│   ├── T1/
│   │   └── <case>-T1.nii.gz
│   ├── DEC/
│   │   └── <case>-DEC.nii.gz
│   ├── GT/
│   │   └── <case>-label.nii.gz
│   ├── splits_final_train.json
│   └── splits_final_test.json
└── CNsEMD_weights/
    └── PHMNet/
        ├── fold_0/BEST_MODEL.pth
        ├── fold_1/BEST_MODEL.pth
        ├── fold_2/BEST_MODEL.pth
        ├── fold_3/BEST_MODEL.pth
        └── fold_4/BEST_MODEL.pth
    └── CNTSeg_V1/
        ├── fold_0/BEST_MODEL.pth
        ├── ...
    └── CNTSegV2_Dedicated/
        ├── fold_0/BEST_MODEL.pth
        ├── ...
```

File prefixes identify the field strength. For example,
`3_100206-T1.nii.gz`, `3_100206-DEC.nii.gz`, and
`3_100206-label.nii.gz` belong to the same 3T case.

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

- train/val set: 162 cases, including 82 at 3T, 40 at 5T, and 40 at 7T.
- test set: 40 cases, including 20 at 3T, 10 at 5T, and 10 at
  7T.
- The train/val set is divided into five training/validation folds.
- The 3T, 5T, and 7T results are subgroup analyses of the same 40-case test
  set.

The fixed partitions are stored in `splits_final_train.json` and
`splits_final_test.json`.

# PHM-Net

## Repository structure

```text
CNsEMD_code/
├── CNsEMD_preprocessing_data.py  # crop volumes and create 2D slices
├── CNsEMD_dataloader.py          # T1w/DEC training dataset
├── CNsEMD_mutilloss.py           # training losses and utilities
├── CNsEMD_train.py               # five-fold training and validation
├── CNsEMD_test.py                # batch ensemble inference and evaluation
├── predict_single_case.py        # inference for one T1w--DEC subject
├── summarize_357T_metrics.py     # mixed, 3T, 5T, and 7T summaries
├── utils.py
└── NetModel/
    ├── PHMNet.py
    ├── CNTSeg_V1.py
    └── CNTSegV2_Dedicated.py
```

## Installation

Create an environment:

```bash
conda create -n cnsemd python=3.9
conda activate cnsemd
```

Install PyTorch and torchvision for the CUDA version available on your
machine by following th official PyTorch installation guide.

Then install the remaining dependencies:

```bash
pip install numpy nibabel SimpleITK scipy scikit-image tqdm thop
```

Run commands from the repository root so that the local `NetModel` package
and `utils.py` can be imported.

## Quick start: single-case inference

Single-case inference requires only a co-registered T1w image and DEC image.
The DEC NIfTI file must have channel-last shape `(X, Y, Z, 3)`, and its
spatial shape and affine must match the T1w image.

Place the downloaded weight folder beside `predict_single_case.py`:

Then run:

```bash
python predict_single_case.py --result-root /path/to/CNsEMD_weights --t1 /path/to/case-T1.nii.gz --dec /path/to/case-DEC.nii.gz
```

Defaults:

- model: `PHMNet`;
- ensemble: folds 0–4;
- device: `cuda:0` when CUDA is available, otherwise CPU;
- output: `<T1-stem>_DEC-<model>-prediction.nii.gz` beside the input T1w image.

All settings except `--t1`, `--dec`, and `--result-root` are optional. For example:

```bash
python predict_single_case.py \
  --t1 /path/to/case-T1.nii.gz \
  --dec /path/to/case-DEC.nii.gz \
  --model CNTSegV2_Dedicated \
  --result-root /path/to/CNsEMD_weights \
  --output /path/to/prediction.nii.gz \
  --device cuda:1
```

The script center-crops or pads the inputs to `128 × 160 × 128`, applies
the same slice-wise T1w normalization used during evaluation, averages the
selected fold logits, and restores the label map to the original T1w space.
It does not perform T1w--DEC registration or generate DEC images from raw
diffusion MRI.

For `CNTSeg_V1_T1DEC`, the weight directory must additionally contain the
two pretrained modality branches:

```text
CNsEMD_weights/CNTSeg_V1_T1DEC/
├── UnetT1/fold_<n>/BEST_MODEL.pth
├── UnetDEC/fold_<n>/BEST_MODEL.pth
└── fold_<n>/BEST_MODEL.pth
```

## Data preprocessing

The preprocessing script center-crops each volume to `128 × 160 × 128`,
clips T1w intensities to the 1st–99th percentiles, applies slice-wise z-score
normalization, preserves the DEC values, and saves labeled axial slices for
training. Foreground training slices are augmented by flipping along the
first spatial axis.

> **Warning:** if `--output_root` already exists, the script deletes that
> directory before regenerating it. Do not select a directory containing
> files that must be retained.

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
| `--num_workers` | `4` | DataLoader workers |
| `--epochs` | `200` | Maximum epochs |
| `--lr` | `0.002` | Initial learning rate |
| `--folds` | all folds | Comma-separated fold names |

Training uses Adam, `ReduceLROnPlateau`, and early stopping with a patience
of 20 epochs. Each fold writes its best checkpoint and training log to:

```text
<result_root>/<model>/fold_<n>/
├── BEST_MODEL.pth
├── training_log.txt
└── Predictions/
```

## Batch ensemble inference and evaluation

`CNsEMD_test.py` loads all five `BEST_MODEL.pth` checkpoints, averages
their logits, reconstructs predictions in the original 3D space, and
optionally evaluates them against expert labels.

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

The reported metrics are Dice, Jaccard index (Jac), average surface
distance (ASD), and average Hausdorff distance (AHD), both across all
foreground classes and for each CN category.

The mixed, 3T, 5T, and 7T summaries can be generated with:

```bash
python summarize_357T_metrics.py \
  --input /path/to/ensemble_metrics_summary.txt
```

## Reproducibility notes

- T1w and DEC volumes must be spatially aligned and use consistent affine
  information.
- DEC volumes must use channel-last layout `(X, Y, Z, 3)`.
- Training and batch evaluation use the fixed five-fold split files.
- Five-fold ensemble inference requires checkpoints for `fold_0` through
  `fold_4`.

## Get DEC images from your ori DWI data using MRtrix3

```
	mrconvert /path/to/data.nii.gz /path/to/DWI.mif -fslgrad /path/to/bvecs /path/to/bvals
    
	dwi2tensor /path/to/DWI.mif /path/to/tensor.mif
    
	tensor2metric /path/to/tensor.mif -vector /path/to/DEC.nii.gz
```


## Citation

If you use CNsEMD or PHM-Net, please cite the associated paper and dataset.
The final paper citation and BibTeX entry will be added after acceptance.


## License

The code and data licences will be specified before the public release.
Users must also comply with the terms associated with the source MRI
datasets and the ScienceDB record.

## Contact

For questions about the code or dataset, please open a GitHub issue or send an email to leix@zjut.edu.cn.
