# [CVPR2025] LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging

This repository will contain the official implementation of our paper:

**LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging**

## Paper
Our paper introduces **LesionLocator**, a novel framework for universal tumor segmentation and tracking in 3D whole-body imaging. Leveraging a large-scale lesion dataset and combining promptable segmentation and deep-learning based registration, our method achieves state-of-the-art performance across both tasks.

> **Authors**: Maximilian Rokuss, Yannick Kirchhoff, Seval Akbal, Balint Kovacs, Saikat Roy, Constantin Ulrich, Tassilo Wald, Lukas T. Rotkopf, Heinz-Peter Schlemmer and Klaus Maier-Hein  
> **Preprint**: [![arXiv](https://img.shields.io/badge/arXiv-2502.20985-b31b1b.svg)](https://arxiv.org/abs/2502.20985)


## Code & Data Release
We've been releasing a lot of code in the past month so I'm sorry for the slight delay here! However, the code for **LesionLocator** will be released very soon. Stay tuned!

In the mean time, if you are interested check out [**nnInteractive**](https://github.com/MIC-DKFZ/nnInteractive) or [**LongiSeg**](https://github.com/MIC-DKFZ/LongiSeg) which we both just released:

- [**nnInteractive**](https://github.com/MIC-DKFZ/nnInteractive) is our recent model for 3D interactive segmentation across all modalities, all target structures using various types of prompts! Check out the [python backend](https://github.com/MIC-DKFZ/nnInteractive) or the [napari plugin](https://github.com/MIC-DKFZ/napari-nninteractive). &nbsp; &nbsp; [![arXiv](https://img.shields.io/badge/arXiv-2503.08373-b31b1b.svg)](https://arxiv.org/abs/2503.08373)

- [**LongiSeg**](https://github.com/MIC-DKFZ/LongiSeg) is an extension of the popular [nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet), designed specifically for **longitudinal medical image segmentation**. By incorporating temporal information across multiple timepoints, LongiSeg enhances segmentation accuracy and consistency, making it a robust tool for analyzing medical imaging over time. LongiSeg includes several methods for temporal feature merging, including the newly introduced [Difference Weighting Block](https://github.com/MIC-DKFZ/Longitudinal-Difference-Weighting). &nbsp; &nbsp;   [![arXiv](https://img.shields.io/badge/arXiv-2404.03010-B31B1B.svg)](https://arxiv.org/abs/2409.13416)

## Citation
If you find our work useful, please consider citing:
```bibtex
@article{rokuss2025lesionlocator,
      title={LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging}, 
      author={Maximilian Rokuss and Yannick Kirchhoff and Seval Akbal and Balint Kovacs and Saikat Roy and Constantin Ulrich and Tassilo Wald and Lukas T. Rotkopf and Heinz-Peter Schlemmer and Klaus Maier-Hein},
      year={2025},
      eprint={2502.20985},
      url={https://arxiv.org/abs/2502.20985}, 
}
```

## Contact
For questions or collaborations, please reach out to maximilian.rokuss@dkfz-heidelberg.de
