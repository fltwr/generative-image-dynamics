# An implementation of Generative Image Dynamics

An implementation of the diffusion model that generates oscillatory motion for an input image introduced in Generative Image Dynamics [1] and a model that synthesizes video frames from a starting frame and optical flow based on softmax splatting [2].

Requirements:

* Python 3.10.12
* torch 2.3.0
* torchvision 0.18.0
* diffusers 0.28.0
* torchmetrics 1.4.0
* opencv-python 4.8.0
* scipy 1.11.4
* matplotlib 3.7.1
* moviepy 1.0.3 (for writing video files)
* cupy 12.2.0 (for softmax splatting)
* pyflow https://github.com/pathak22/pyflow (for optical flow estimation)

Download trained models:

* https://1drv.ms/u/s!AjGGQwItv34-bK738lmdo7wf2uk?e=cWvbXo

References:

* [1] Zhengqi Li, Richard Tucker, Noah Snavely, and Aleksander Holynski. Generative image dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
* [2] Simon Niklaus and Feng Liu. Softmax splatting for video frame interpolation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5437–5446, 2020.

The images and videos in this repo are from Wikimedia Commons.

## Generate optical flow from image

Following the paper the motion synthesis model is implemented as a latent diffusion model which consists of a variational autoencoder (VAE) and a U-Net. The VAE was taken from [CompVis/ldm-celebahq-256](https://huggingface.co/CompVis/ldm-celebahq-256) and the U-Net was trained from scratch. The frequency attention layers were not implemented.

Example:

![](data/unet_sample.gif)

* OpenCV remapping is used for warping in the above example.

## Generate video from image and optical flow

A frame synthesis model takes an image and a forward flow field to predict a warped image. This model was not implemented following [1], but modified from the model in [2] which uses softmax splatting to warp image features at different resolutions and a GridNet to generate an image from warped features. The model in [1] also uses softmax splatting and a feature pyramid, but the output image is generated by the synthesis network from co-modulation GAN.

Evaluation:

| Method | PSNR &#8593; | SSIM &#8593; | LPIPS (AlexNet) &#8595; |
| --- | --- | --- | --- |
| Model | 36.3127 | 0.9720 | 0.0096 |
| Average splatting | 34.8256 | 0.9657 | 0.0236 |
| OpenCV remapping | 34.7156 | 0.9654 | 0.0132 |

* The model is compared with two baselines average splatting [2] and OpenCV remapping with bicubic interpolation. The model warps multi-scale features and while the other methods warp the image directly.
* Evaluation data: 82 frame sequences, 150 frames per sequence, resolution 256x160.

Example:

![](data/frame_synthesis_sample.gif)
