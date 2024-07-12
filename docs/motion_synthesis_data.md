# Notes on motion synthesis training data

Summary:

1. I used 818 short videos for training. One or more 150-frame sequences were taken from each video at frame rate 30 and resolution 256 (width) x 160 (height), resulting in 7218 sequences in total.
2. The contents of these videos are limited to plants moving in the wind. I collected them from YouTube by searching plant names, places ("mountain", "meadow", " arboretum"...), seasons, "wind", "tripod" (to avoid camera motion), "no looping" (to avoid duplicates), etc.
3. Optical flow were estimated using [pyflow](https://github.com/pathak22/pyflow) with [these parameters](../utils/flow.py#L7). For each sequence, the optical flow from the first frame to the other frames (rather than between consecutive frames) are needed for training the U-Net.
4. FFT were normalized for the diffusion model by scaling the coefficients with frequency-specific factors, which are based on training data statistics [data/labels/fft_std.npy](../data/labels/fft_std.py).

## Collecting videos

The collected videos were cut into short videos that are single camera shots. These short videos were filtered to remove duplicates and camera motion.

## Optical flow estimation

The training code assumes that the estimations are stored as NPY files. I stored them in half-precision to save space.

Sequences with median optical flow magnitude larger than 0.5 were removed (excluded from [data/labels/motion_synthesis_train_set.csv](../data/labels/motion_synthesis_train_set.csv)), to filter out incorrect optical flow estimations under this setting.

## Normalizing FFT

The paper applied "frequency adaptive normalization" to the FFT of optical flow, which is to first divide the coefficients at each frequency by the 95th percentile of the magnitudes and then take square-root while keeping the sign.

I used scaled standard deviations instead of percentiles for scaling and omitted the square-root transformation. At each frequency, FFT coefficients were divided by the standard deviation from zero calculated on the training set and by a `scale` parameter. The standard deviations were saved in [data/labels/fft_std.npy](../data/labels/fft_std.py) and `scale` was set to 2.82, so that around 3% of the coefficients were out of the range [-1, 1] after normalization.

`FrameSpectrumDataset` in [utils/dataset.py](../utils/dataset.py) with methods `get_std_from_zero` and `test_scales` can be used if you wish to recalculate the standard deviations or adjust the `scale` parameter.

## Contents of training videos

I had 818 training videos with at least one sequence that has valid optical flow estimation. They are videos of plants gently moving in the wind. Below are the first frames of 64 videos randomly sampled from the set.

![image](https://github.com/fltwr/generative-image-dynamics/assets/45270448/4de05c55-5ed4-45c2-81af-f821d6b51286)
![image](https://github.com/fltwr/generative-image-dynamics/assets/45270448/f7b9dcf8-06a7-4823-b4b4-07b2cdc9ef64)
![image](https://github.com/fltwr/generative-image-dynamics/assets/45270448/436b1552-bac7-4998-a646-a8430722a2f0)
![image](https://github.com/fltwr/generative-image-dynamics/assets/45270448/a2eb6bee-8e45-4de1-a10f-e83a5ea2bdd2)
![image](https://github.com/fltwr/generative-image-dynamics/assets/45270448/14f2d371-043f-4fea-bf04-19b9d5686c60)
![image](https://github.com/fltwr/generative-image-dynamics/assets/45270448/33b486ef-056a-43ad-96eb-ba952f1becd9)
![image](https://github.com/fltwr/generative-image-dynamics/assets/45270448/66c11d38-390a-4f8d-ad2f-962a40a8e75c)
![image](https://github.com/fltwr/generative-image-dynamics/assets/45270448/588c0645-4f95-4da7-b6ab-cdf29f138149)



