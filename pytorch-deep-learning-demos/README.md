# PyTorch Deep Learning Demos

This module contains GUI-based deep learning demos for image classification and image generation.

## Technical Scope

### VGG19-BN Classification Demo

- CIFAR-10 image classification
- Data augmentation visualization
- Model structure display
- Training curve visualization
- Single-image inference

### DCGAN Generation Demo

- MNIST-style image generation
- Generator and discriminator model implementation
- Training loss visualization
- Real and generated image comparison

## Technologies

- Python
- PyTorch
- Torchvision
- PyQt
- Matplotlib
- NumPy

## Project Structure

```text
pytorch-deep-learning-demos/
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
├─ loss_plot.png
└─ requirements
```

## Run

### VGG19-BN Classification GUI

```bash
cd pytorch-deep-learning-demos
python main1.py
```

### DCGAN Generation GUI

```bash
cd pytorch-deep-learning-demos
python main2.py
```

## External Files

Trained checkpoints and demo image archives are excluded from this repository due to file size.

Required files for full inference:

```text
pytorch-deep-learning-demos/best_vgg19_bn.pth
pytorch-deep-learning-demos/weights/netG_epoch_50.pth
pytorch-deep-learning-demos/Q1_image/
pytorch-deep-learning-demos/Q2_images/
```

Checkpoint usage:

| Demo | Required Checkpoint |
|---|---|
| VGG19-BN classification | `best_vgg19_bn.pth` |
| DCGAN generation | `weights/netG_epoch_50.pth` |

The checkpoints can also be reproduced by running:

```bash
python trainQ1.py
python trainQ2.py
```

Training may take a long time without a GPU.