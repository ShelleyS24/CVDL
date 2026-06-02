# Computer Vision and Deep Learning Projects

This repository contains two GUI-based computer vision projects implemented with OpenCV, PyQt, and PyTorch.

The projects cover both classical computer vision methods and deep learning-based vision models, including camera calibration, stereo vision, feature matching, CNN classification, and GAN-based image generation.

## Project Modules

| Module | Description | Main Technologies |
|---|---|---|
| `opencv-computer-vision-demos` | Classical computer vision demos with GUI interaction | OpenCV, PyQt, NumPy |
| `pytorch-deep-learning-demos` | Deep learning vision demos for classification and generation | PyTorch, Torchvision, PyQt |

## Highlights

- Built PyQt-based GUI interfaces for visualizing computer vision workflows.
- Implemented classical OpenCV tasks including calibration, AR projection, stereo disparity, and SIFT matching.
- Implemented PyTorch-based VGG19-BN classification and DCGAN image generation demos.
- Organized model definitions, training scripts, inference scripts, and result visualizations into a reproducible project structure.
- Excluded large datasets and model checkpoints from Git tracking to keep the repository lightweight.

## Repository Structure

```text
CVDL/
├─ opencv-computer-vision-demos/
│  ├─ main
│  ├─ question1.py
│  ├─ question2.py
│  ├─ question3.py
│  ├─ question4.py
│  └─ qtUI.ui
│
└─ pytorch-deep-learning-demos/
   ├─ main1.py
   ├─ main2.py
   ├─ Q1.py
   ├─ Q2.py
   ├─ trainQ1.py
   ├─ trainQ2.py
   ├─ models/
   │  ├─ VGG19_BN.py
   │  └─ DCGAN.py
   ├─ training_validation_metrics.png
   └─ loss_plot.png
```

## Setup

The original projects were developed with Python 3.8.

Install dependencies:

```bash
pip install -r opencv-computer-vision-demos/requirements
pip install -r pytorch-deep-learning-demos/requirements
```

If needed, install the main packages directly:

```bash
pip install opencv-python opencv-contrib-python numpy PyQt5 matplotlib torch torchvision torchsummary Pillow
```

## Run

### OpenCV Computer Vision Demos

```bash
cd opencv-computer-vision-demos
python main
```

### PyTorch VGG19-BN Classification Demo

```bash
cd pytorch-deep-learning-demos
python main1.py
```

### PyTorch DCGAN Generation Demo

```bash
cd pytorch-deep-learning-demos
python main2.py
```

## External Files

Large image archives, downloaded datasets, and trained model checkpoints are not included in this repository.

Download the external files here:

```text
Google Drive: https://drive.google.com/file/d/1khE8YDUN16FRhxdJydqmi9cH8a0lcpBj/view?usp=drive_link
```

Required external files:

```text
pytorch-deep-learning-demos/best_vgg19_bn.pth
pytorch-deep-learning-demos/weights/netG_epoch_50.pth
pytorch-deep-learning-demos/Q1_image/
pytorch-deep-learning-demos/Q2_images/
```

After downloading, place the files under the project root using the following structure:

```text
CVDL/
├─ opencv-computer-vision-demos/
│  ├─ Image.zip
└─ pytorch-deep-learning-demos/
   ├─ best_vgg19_bn.pth
   ├─ Q1_image.zip
   ├─ Q2_images.zip
   └─ weights/
      └─ netG_epoch_50.pth
```

Then extract:

```text
opencv-computer-vision-demos/Image.zip    ->  opencv-computer-vision-demos/Q1_image/
                                              opencv-computer-vision-demos/Q2_image/
                                              opencv-computer-vision-demos/Q3_image/
                                              opencv-computer-vision-demos/Q4_image/
                                              opencv-computer-vision-demos/Q5_image/
                                              
pytorch-deep-learning-demos/Q1_image.zip  ->  pytorch-deep-learning-demos/Q1_image/
pytorch-deep-learning-demos/Q2_images.zip ->  pytorch-deep-learning-demos/Q2_images/
```

Without these files, the GUI windows can still be opened, but some demo or inference functions may not run.

## Notes

This repository is intended as a portfolio-style organization of computer vision and deep learning implementations.  
Generated files, model checkpoints, cache files, downloaded datasets, and large archives are excluded from Git tracking.