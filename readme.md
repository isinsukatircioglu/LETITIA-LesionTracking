## **LETITIA: Learning Tumor Dynamics and Early Markers of Immunotherapy Response from PET/CT Imaging** 

This repository builds upon and adapts the code from:

 **LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging**

The paper introduces a novel framework for **zero-shot lesion segmentation** and **longitudinal tumor tracking** in 3D full-body imaging. By combining a large-scale lesion dataset, promptable segmentation, and deep-learning-based image registration, our framework achieves state-of-the-art results for both tasks.

> **Authors**: Maximilian Rokuss, Yannick Kirchhoff, Seval Akbal, Balint Kovacs, Saikat Roy, Constantin Ulrich, Tassilo Wald, Lukas T. Rotkopf, Heinz-Peter Schlemmer and Klaus Maier-Hein  
> **Paper**: [![CVPR](https://img.shields.io/badge/%20CVPR%202025%20-open%20access-blue.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Rokuss_LesionLocator_Zero-Shot_Universal_Tumor_Segmentation_and_Tracking_in_3D_Whole-Body_CVPR_2025_paper.html)

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/MIC-DKFZ/LesionLocator
cd LesionLocator
```

### 2. Create a conda environment (recommended)

```bash
conda create -n lesionlocator python=3.12 -y
conda activate lesionlocator
```

### 3. Install the package

```bash
pip install -e .
```

---

## 🚀 Features & Usage

LETITIA LesionLocator has **two main modes**:

### 1️⃣ Zero-Shot Lesion Segmentation (Single Timepoint)

Perform lesion segmentation using **point** or **3D bounding box** prompts. No lesion type–specific fine-tuning required.

#### Usage

```bash
LesionLocator_track
    -i /path/to/image(s)
    -p /path/to/groundtruth masks(s) 
    -t point 
    -o /path/to/output/folder
    -m /path/to/LesionLocatorCheckpoint
    -npp 2
    -nps 2
    --modality "ct"
    --visualize
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `-i` | **Input image(s)**: Path to an image file (`.nii.gz`) or a folder containing multiple images. |
| `-p` | **Prompt(s)**: Can be `.nii.gz` instance segmentation maps. Prompts must have the same filename as their corresponding input image. Binary masks will be automatically converted to instance maps.
| `-t` | **Prompt type**: Choose between `point` or `box`. Determines how the model interprets the prompts. Default is `box`. |
| `-o` | **Output folder**: Path where the predicted segmentation masks will be saved. Created automatically if it doesn't exist. |
| `-m` | **Model folder**: Path to the downloaded `LesionLocatorCheckpoint` directory containing trained model weights. |
| `-f` | **Model folds**: Specify one or more folds. Defaults to all 5 folds for ensemble prediction. |
| `--disable_tta` | Disables test-time augmentation (TTA) using mirroring. Speeds up inference at the cost of accuracy. |
| `--continue_prediction`, `--c` | Continues a previously interrupted run by skipping existing output files. |
| `--visualize` | Saves axial and coronal views of the predicted and ground truth masks.|


🧠 If you provide a 3D (instance) segmentation mask as a prompt, LesionLocator will internally extract the bounding boxes or points automatically, depeding on the specified prompt type `-t`. Details on how to handle promting and the .json format can be found [here](/documentation/prompting.md).

**Examples:**

```bash
LesionLocator_track -i /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr -p /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr -m /scratch/LesionLocatorckpt/LesionLocatorCheckpoint -o /home/katircio/code/LesionLocator/Seg801 -t "point" -npp 2 -nps 2 --visualize --modality "ct" 
```
---

### 2️⃣ Lesion Segmentation & Tracking Across Timepoints (Longitudinal)

Includes the lesion segmentation above but also tracks lesions across multiple timepoints using previous labels or prompts when available.

#### Usage

```bash
LesionLocator_track
    -i /path/to/image(s)
    -p /path/to/groundtruth masks(s) 
    -t point 
    -o /path/to/output/folder
    -m /path/to/LesionLocatorCheckpoint
    -npp 2
    -nps 2
    --modality "ct"
    --visualize
    --adaptive_mode
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `-i` | **Input image(s)**: Path to an image file (`.nii.gz`) or a folder containing multiple images. |
| `-p` | **Prompt(s)**: Can be `.nii.gz` instance segmentation maps. Prompts must have the same filename as their corresponding input image. Binary masks will be automatically converted to instance maps.
| `-t` | **Prompt type**: Choose between `point` or `box`. Determines how the model interprets the prompts. Default is `box`. |
| `-o` | **Output folder**: Path where the predicted segmentation masks will be saved. Created automatically if it doesn't exist. |
| `-m` | **Model folder**: Path to the downloaded `LesionLocatorCheckpoint` directory containing trained model weights. |
| `-f` | **Model folds**: Specify one or more folds. Defaults to all 5 folds for ensemble prediction. |
| `--disable_tta` | Disables test-time augmentation (TTA) using mirroring. Speeds up inference at the cost of accuracy. |
| `--continue_prediction`, `--c` | Continues a previously interrupted run by skipping existing output files. |
| `--visualize` | Saves axial and coronal views of the predicted and ground truth masks.|
| `--track` |Set this flag to enable tracking. This will use the LesionLocatorTrack model to track lesions.|
| `--adaptive_mode` | Enable selection between segmentation and tracking based on Dice/NSD scores.|

**Examples**

```bash
LesionLocator_track -i /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr -p /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr -m /scratch/LesionLocatorckpt/LesionLocatorCheckpoint -o /home/katircio/code/LesionLocator/ResultSegTrack801aux -t "point" -npp 2 -nps 2 --visualize --modality "ct" --track --adaptive_mode

```

To run inference with LesionLocator, you need to download the pretrained model checkpoint: 

#### [🔗 Download LesionLocator Checkpoint](https://zenodo.org/records/15174217)

Once downloaded, extract the contents and use the `-m` argument in the CLI tools to point to the directory named `LesionLocatorCheckpoint`. Let us know if you run into issues downloading or loading checkpoints!

---
## Inference on RunAI for USZ Melanoma Dataset
```bash
runai submit --name seg -i registry.rcp.epfl.ch/letitia/my-pytorch:v1 --gpu 0.5 --memory 60G --memory-limit 75G --large-shm  --pvc letitia-scratch:/scratch --pvc home:/home/katircio --command -- /bin/bash -ic 'set -ex; echo "Starting job"; conda activate lesionlocator ;cd /home/katircio/code/LesionLocator/ ; LesionLocator_track -i /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr -p /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr -m /scratch/LesionLocatorckpt/LesionLocatorCheckpoint -o /home/katircio/code/LesionLocator/LesionSegUSZ801 -t "point" -npp 1 -nps 1 --visualize --modality "ct"'
```

```bash
runai submit --name track -i registry.rcp.epfl.ch/letitia/my-pytorch:v1 --gpu 0.5 --memory 60G --memory-limit 75G --large-shm  --pvc letitia-scratch:/scratch --pvc home:/home/katircio --command -- /bin/bash -ic 'set -ex; echo "Starting job"; conda activate lesionlocator ;cd /home/katircio/code/LesionLocator/ ; LesionLocator_track -i /scratch/nnUNet_raw/Dataset801_USZMelanoma/imagesTr -p /scratch/nnUNet_raw/Dataset801_USZMelanoma/labelsTr -m /scratch/LesionLocatorckpt/LesionLocatorCheckpoint -o /home/katircio/code/LesionLocator/LesionTrackUSZ801 -t "point" -npp 1 -nps 1 --visualize --modality "ct" --track --adaptive_mode'
```
---

## 📚 Citation

If you use this code, please cite the original paper:

```bibtex
@InProceedings{Rokuss_2025_CVPR,
    author    = {Rokuss, Maximilian and Kirchhoff, Yannick and Akbal, Seval and Kovacs, Balint and Roy, Saikat and Ulrich, Constantin and Wald, Tassilo and Rotkopf, Lukas T. and Schlemmer, Heinz-Peter and Maier-Hein, Klaus},
    title     = {LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {30872-30885}
}
```
