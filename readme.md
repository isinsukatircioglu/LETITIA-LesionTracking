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

If you want to use the napari viewer for visual inspection:

```bash
pip install "napari[all]"
```

---

## 🚀 Features & Usage

LesionLocator has **two main modes**:

### 1️⃣ Zero-Shot Lesion Segmentation (Single Timepoint)

Perform universal lesion segmentation using **point** or **3D bounding box** prompts. No lesion type–specific fine-tuning required.

#### Usage

```bash
LesionLocator_segment
    -i /path/to/image(s)
    -p /path/to/prompt(s) 
    -t box 
    -o /path/to/output/folder
    -m /path/to/LesionLocatorCheckpoint 
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `-i` | **Input image(s)**: Path to an image file (`.nii.gz`) or a folder containing multiple images. |
| `-p` | **Prompt(s)**: Can be either `.json` files containing 3D point or box coordinates, or `.nii.gz` instance segmentation maps. Prompts must have the same filename as their corresponding input image. Binary masks will be automatically converted to instance maps.  More info [here](/documentation/prompting.md). |
| `-t` | **Prompt type**: Choose between `point` or `box`. Determines how the model interprets the prompts. Default is `box`. |
| `-o` | **Output folder**: Path where the predicted segmentation masks will be saved. Created automatically if it doesn't exist. |
| `-m` | **Model folder**: Path to the downloaded `LesionLocatorCheckpoint` directory containing trained model weights. |
| `-f` | **Model folds**: Specify one or more folds. Defaults to all 5 folds for ensemble prediction. |
| `--disable_tta` | Disables test-time augmentation (TTA) using mirroring. Speeds up inference at the cost of accuracy. |
| `--continue_prediction`, `--c` | Continues a previously interrupted run by skipping existing output files. |
| `--visualize` | Opens results in a `napari` viewer for inspection. Requires `napari[all]` installed. |


🧠 If you provide a 3D (instance) segmentation mask as a prompt, LesionLocator will internally extract the bounding boxes or points automatically, depeding on the specified prompt type `-t`. Details on how to handle promting and the .json format can be found [here](/documentation/prompting.md). You can also run `LesionLocator_segment -h` for help. 

**Examples:**

```bash
LesionLocator_segment -i image.nii.gz -p label.nii.gz -t box -o /output/folder -m /path/to/LesionLocatorCheckpoint 
```
```bash
LesionLocator_segment -i image.nii.gz -p points.json -t point -o /output/folder -m /path/to/LesionLocatorCheckpoint 
```
```bash
LesionLocator_segment -i /image/folder -p /prompts/folder -t box -o /output/folder -m /path/to/LesionLocatorCheckpoint 
```

---

### 2️⃣ Lesion Segmentation & Tracking Across Timepoints (Longitudinal)

Includes the lesion segmentation above but also tracks lesions across multiple timepoints using previous labels or prompts.

#### Usage

```bash
LesionLocator_track
    -bl /path/to/baseline.nii.gz
    -fu /path/to/followup1.nii.gz /path/to/followup2.nii.gz
    -p /path/to/baseline_prompt_or_mask(s)
    -t prev_mask
    -o /path/to/output
    -m /path/to/LesionLocatorCheckpoint
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `-bl` | **Baseline scan**: Path to the `.nii.gz` image used as the baseline. Must match format expected by the model (or dataset config). |
| `-fu` | **Follow-up scan(s)**: One or more `.nii.gz` files representing follow-up timepoints. Provide multiple paths separated by space for autoregressive tracking. |
| `-p` | **Prompt**: Path to the baseline prompt. Can be one `.nii.gz` instance segmentation mask or list of several semantic segmentation masks. Can also be a `.json` file with a point/box prompt. Used to initialize tracking. More info [here](/documentation/prompting.md). |
| `-o` | **Output folder**: Where tracked lesion segmentations will be saved. Created automatically if it doesn't exist. |
| `-t` | **Prompt type**: Options: `point`, `box`, or `prev_mask`. Use `point`/`box` to trigger segmentation of baseline lesions before tracking. Use `prev_mask` to track from an existing baseline label mask directly. Default is `prev_mask`. |
| `-m` | **Model folder**: Path to the `LesionLocatorCheckpoint` directory containing tracking model weights. |
| `-device` | **Device selection**: Options are `cuda`, `cpu`, or `mps` (Apple Silicon). Controls where the inference runs. Default is `cuda`. |

You can also run `LesionLocator_track -h` for help.

**Examples**

This will do both the segmentation of the baseline scan using box prompts generated from `baseline_label.nii.gz` and then track each lesion:
```bash
LesionLocator_track -bl baseline.nii.gz -fu followup.nii.gz -p baseline_label.nii.gz -t box -o output/folder -m /path/to/LesionLocatorCheckpoint
```

Similarly works with `.json` prompts:
```bash
LesionLocator_track -bl baseline.nii.gz -fu followup.nii.gz -p points.json -t point -o output/folder -m /path/to/LesionLocatorCheckpoint
```

This will trigger *just the tracking* (`-t prev_mask`) of all provided semantic masks autoregressively for all follow-ups:
```bash
LesionLocator_track -bl baseline.nii.gz -fu followup1.nii.gz followup2.nii.gz -p mask1.nii.gz mask2.nii.gz mask3.nii.gz -t prev_mask -o output/folder -m /path/to/LesionLocatorCheckpoint
```

 To run inference with LesionLocator, you need to download the pretrained model checkpoint: 

#### [🔗 Download LesionLocator Checkpoint](https://zenodo.org/records/15174217)

Once downloaded, extract the contents and use the `-m` argument in the CLI tools to point to the directory named `LesionLocatorCheckpoint`. Let us know if you run into issues downloading or loading checkpoints!

---

## 📚 Citation

If you find our work useful, please consider citing:

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
