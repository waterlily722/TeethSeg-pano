---
title: "Instance Segmentation of Teeth in Panoramic X-rays"
author: "ECE3070 Medical Image Analysis Coursework"
date: "May 2026"
---

# Instance Segmentation of Teeth in Panoramic X-rays: Original OralBBNet and Modified OralBBNet

## Cover Page

**Course:** ECE3070 Medical Image Analysis

**Project title:** Instance Segmentation of 32 FDI Tooth Positions in Panoramic Dental X-rays

**Team information:**

- Name 1, Student ID 1, Class 1
- Name 2, Student ID 2, Class 2

**Member contributions:**

- Member A: U-Net and OralBBNet method development, model training, hyperparameter configuration, quantitative evaluation, and result analysis.
- Member B: data preprocessing, dataset indexing, Active Contour baseline, YOLO bounding-box label parsing, visualization, and report writing.

## 1. Overview

This project studies tooth instance segmentation in panoramic dental X-rays. The task is to predict 32 permanent tooth-position masks following the FDI numbering system. Compared with ordinary binary tooth segmentation, this is harder because the model must both locate tooth pixels and assign them to the correct tooth channel.

Two experimental settings are used in the final comparison:

- **Reference OralBBNet**: the reproduction-oriented baseline that keeps the original OralBBNet-style training configuration as much as possible, including softmax output, Dice+L2 loss, Adam learning rate `3e-4`, dropout `0.12`, 60 epochs, global batch size 2, and fixed threshold `0.5`.
- **Modified OralBBNet**: the proposed method that uses a corrected BCE+DiceLoss objective, sigmoid output, a more stable learning rate, fixed threshold `0.5` for the main comparison, and an additional threshold-calibration analysis.

The main evaluation in this report uses **threshold = 0.5** for both reference and modified experiments. This keeps the comparison aligned with the reference evaluation protocol and avoids over-reporting results from very low tuned thresholds.

The report therefore answers three questions:

1. What was changed from Reference OralBBNet to Modified OralBBNet?
2. How much improvement is observed at the fixed threshold `0.5`?
3. Why does OralBBNet have an advantage over U-Net in terms of segmentation quality, efficiency of supervision, and practical trade-offs?

## 2. Data and Preprocessing

The experiments use the UFBA-425 panoramic X-ray dataset. Each image is converted to grayscale, enhanced with CLAHE, resized to `512 x 512`, and paired with a 32-channel target mask. Each output channel corresponds to one FDI tooth position:

```text
11-18, 21-28, 31-38, 41-48
```

YOLO-format bounding-box labels are converted into 32-channel spatial prior maps. These prior maps are used only by OralBBNet. U-Net receives only the X-ray image, while OralBBNet receives the concatenation of bounding-box priors and the image.

The two experimental settings use slightly different split logic because they serve different purposes. The reference setting follows the reference configuration more closely: it forms a train pool and test set by category and then takes a validation split from the train pool. The modified setting uses a fixed random `70%/15%/15%` train/validation/test split. Therefore, the exact Active Contour numbers are not identical across the two settings. For model-to-model comparison within each setting, however, all methods share the same split.

| Item | Reference setting | Modified setting |
| --- | --- | --- |
| Dataset | UFBA-425 | UFBA-425 |
| Image size | `512 x 512` | `512 x 512` |
| Preprocessing | Grayscale + CLAHE | Grayscale + CLAHE |
| Target | 32 FDI mask channels | 32 FDI mask channels |
| Split | Category-based train/test, then validation from train pool | Random `70/15/15` train/validation/test split |
| Prior source | YOLO-format labels / fallback boxes | YOLO-format labels / fallback boxes |
| Main threshold | `0.5` | `0.5` |

## 3. Compared Methods

### 3.1 Active Contour

Active Contour is used as a traditional image-processing baseline. It does not require training and produces a single binary tooth-region mask. It is useful as a lower-bound reference, but it cannot distinguish tooth identity or output 32 FDI channels.

### 3.2 U-Net

U-Net is the supervised deep-learning baseline. It takes a `512 x 512 x 1` X-ray image and predicts a `512 x 512 x 32` mask tensor. It can learn tooth boundaries from annotations, but it has no explicit information about where each tooth number should appear.

### 3.3 Reference OralBBNet

Reference OralBBNet follows the reference training configuration as closely as possible:

| Component | Reference OralBBNet |
| --- | --- |
| Input | 32 bounding-box prior channels + 1 X-ray channel |
| Output activation | Softmax |
| Loss | Original Dice + L2-style loss |
| Optimizer | Adam, learning rate `3e-4`, beta1/momentum `0.99` |
| Dropout | `0.12` |
| Epochs | 60 |
| Batch setting | 2 global batch size using two GPUs |
| Evaluation threshold | Fixed `0.5` |

One engineering compatibility fix was required: `tf.multiply(skip, prior)` was replaced by `layers.Multiply()([skip, prior])`. This does not change the mathematical operation; it only makes the model build correctly under the current Keras version.

The original loss is kept for reproduction, but it has an important limitation. Dice itself is a score where higher is better, while training minimizes the loss. Therefore, directly minimizing a positive Dice term can make the optimization objective poorly aligned with the final segmentation metric. This is one reason for adding the modified experiment.

### 3.4 Modified OralBBNet

The modified OralBBNet keeps the same high-level idea, but changes the optimization and output design:

| Component | Modified setting |
| --- | --- |
| Input | 32 bounding-box prior channels + 1 X-ray channel |
| Output activation | Sigmoid |
| Loss | `0.2 * BCE + 0.8 * DiceLoss` |
| Optimizer | Adam, learning rate `1e-4`, gradient clipping |
| Dropout | `0.12` |
| Epochs | 60 |
| Batch setting | 2 global batch size using two GPUs |
| Main evaluation threshold | Fixed `0.5` |
| Extra analysis | Validation-set threshold calibration |

The most important modification is the loss. In the original formulation, Dice was used directly inside the minimized objective, even though Dice is a metric that should be maximized. The modified version uses `1 - Dice` and combines it with BCE. This makes the optimization direction consistent with segmentation quality.

The modified model also uses sigmoid output instead of softmax. This is more suitable for 32 independent tooth channels because adjacent teeth are different instances, not a single mutually exclusive semantic class over the whole image.

Table 2 summarizes the exact experimental settings used in the two deep-learning methods.

**Table 2. Original and modified training settings.**

| Setting | Reference OralBBNet | Modified OralBBNet |
| --- | --- | --- |
| Method name | Reference OralBBNet | Modified OralBBNet |
| Role | Reference reproduction baseline | Proposed modified method |
| GPU setting | Two visible GPUs, global batch 2 | Two visible GPUs, global batch 2 |
| Per-GPU batch | 1 | 1 |
| Epochs | 60 | 60 |
| Base filters | 64 | 64 |
| Optimizer | Adam | Adam |
| Learning rate | `3e-4` | `1e-4` |
| Adam beta1 | `0.99` | default Adam beta1 |
| Gradient clipping | Not used | `clipnorm=1.0` |
| LR scheduler | Halve LR after 5 stagnant validation epochs | `ReduceLROnPlateau`, patience 5 |
| Early stopping | Not used | Patience 12, restore best weights |
| Dropout | `0.12` | `0.12` |
| Output activation | Softmax | Sigmoid |
| Loss | Original Dice + L2-style term | `0.2 * BCE + 0.8 * DiceLoss` |
| Main threshold | `0.5` | `0.5` |
| Extra threshold analysis | No | Yes, validation-set calibration |

## 4. Training Setup

Both reference and modified experiments are trained for 60 epochs with fixed threshold `0.5` used for the main reported test metrics. Both methods use two GPUs with one sample per GPU, so the effective global batch size is 2. This keeps the batch setting aligned across U-Net and OralBBNet comparisons.

Reference OralBBNet follows the reference hyperparameters more closely: Adam uses learning rate `3e-4` and beta1 `0.99`, the model output uses softmax, and the loss keeps the original Dice+L2-style objective. Modified OralBBNet lowers the learning rate to `1e-4`, uses sigmoid output, applies gradient clipping, and optimizes a BCE+DiceLoss objective.

The fixed-threshold evaluation is the primary result because it is directly comparable across original and modified methods. A tuned-threshold analysis is also performed for the modified method, but it is treated as supplementary because it changes the decision threshold after validation.

For reproducibility, the reference and modified experimental outputs were kept separate. The analysis below refers to them as methods.

Training histories show that the modified objective is numerically much better behaved. Since the loss definitions are different, the absolute loss values should not be compared directly between original and modified experiments. However, within each experiment the validation curves are meaningful. The best validation losses occurred at:

| Model | Epochs trained | Best validation epoch | Best validation loss |
| --- | ---: | ---: | ---: |
| Original U-Net | 60 | 45 | 210.607 |
| Original OralBBNet | 60 | 58 | 172.236 |
| Modified U-Net | 60 | 59 | 0.320 |
| Modified OralBBNet | 60 | 59 | 0.260 |

The modified models continue improving until late training, but the validation loss is smoother and the scale is easier to interpret because it is based on BCE plus true DiceLoss.

![Validation loss comparison](results_compare/report_validation_loss_comparison.png)

*Figure 1. Validation loss curves for original and modified settings. The loss scales are different because the original and modified objectives are not numerically comparable, but the curves show convergence behavior under each training setup.*

## 5. Fixed-Threshold Results

Table 1 reports the main test-set results using threshold `0.5`.

**Table 1. Test-set performance at threshold 0.5.**

| Experiment | Method | Binary Dice | Binary IoU | Mean tooth/channel Dice |
| --- | --- | ---: | ---: | ---: |
| Original | Active Contour | 0.350 | 0.215 | - |
| Original | U-Net | 0.716 | 0.568 | 0.436 |
| Original | OralBBNet | 0.863 | 0.768 | 0.756 |
| Modified | Active Contour | 0.356 | 0.219 | - |
| Modified | U-Net | 0.894 | 0.811 | 0.826 |
| Modified | OralBBNet | **0.907** | **0.831** | **0.871** |

The metrics have different meanings. Binary Dice and Binary IoU merge all 32 tooth channels into one tooth-versus-background mask. They measure whether the model finds tooth regions. Mean tooth/channel Dice is stricter because it evaluates tooth-position channels. It measures whether the model assigns pixels to the correct FDI tooth identity.

The modified OralBBNet is best on all three main metrics. At fixed threshold `0.5`, it reaches `0.907` Binary Dice and `0.871` mean tooth/channel Dice. This indicates that the modified model improves both region localization and tooth identity assignment.

![Fixed-threshold performance](results_compare/report_fixed_threshold_performance.png)

*Figure 2. Fixed-threshold performance comparison. Modified OralBBNet gives the best overall performance at threshold 0.5.*

## 6. Original OralBBNet vs Modified OralBBNet

The modified OralBBNet improves over the original OralBBNet under the same threshold `0.5`:

| Metric | Original OralBBNet | Modified OralBBNet | Absolute gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| Binary Dice | 0.863 | 0.907 | +0.044 | +5.1% |
| Binary IoU | 0.768 | 0.831 | +0.063 | +8.2% |
| Mean tooth/channel Dice | 0.756 | 0.871 | +0.115 | +15.2% |

The improvement mainly comes from three changes.

First, the loss function is more appropriate. BCE stabilizes pixel-level supervision, while DiceLoss directly optimizes overlap. This is better aligned with the final Dice-based evaluation.

Second, sigmoid output avoids forcing all tooth channels to compete through one softmax distribution. For tooth instance segmentation, multiple nearby channels need independent probability maps, so sigmoid is a more natural output activation.

Third, the modified bounding-box gate uses a sigmoid-based prior modulation. This keeps the spatial prior helpful while avoiding overly aggressive suppression of skip-connection features.

The improvement is especially meaningful for the stricter tooth/channel metric. The mean tooth/channel Dice improves by `0.115`, which is a larger relative gain than Binary Dice. This suggests that the modifications help not only with foreground-background separation, but also with assigning the prediction to the correct tooth channel.

A more detailed tooth-region comparison is:

| Tooth group | Original OralBBNet | Modified OralBBNet | Gain |
| --- | ---: | ---: | ---: |
| Incisors | 0.705 | 0.893 | +0.188 |
| Canines | 0.770 | 0.869 | +0.099 |
| Premolars | 0.752 | 0.880 | +0.128 |
| Molars | 0.787 | 0.907 | +0.120 |

The largest gain is observed for incisors. Incisors are relatively small and close to the image midline, so explicit spatial priors and better channel-wise optimization are helpful.

Qualitatively, the modified result is also more localized and better aligned with the ground truth.

![Original qualitative result](results_original/qualitative_examples.png)

*Figure 3. Qualitative result from the original OralBBNet setting.*

![Modified qualitative result](results_compare/qualitative_examples.png)

*Figure 4. Qualitative result from the modified OralBBNet setting.*

## 7. OralBBNet vs U-Net

### 7.1 Performance

In the original experiment, OralBBNet is much stronger than U-Net:

| Setting | Binary Dice gain | Binary IoU gain | Mean tooth/channel Dice gain |
| --- | ---: | ---: | ---: |
| Original OralBBNet - Original U-Net | +0.147 | +0.200 | +0.320 |

In the modified experiment, U-Net is already much stronger because of the improved training setup. Even so, OralBBNet still improves the main metrics:

| Setting | Binary Dice gain | Binary IoU gain | Mean tooth/channel Dice gain |
| --- | ---: | ---: | ---: |
| Modified OralBBNet - Modified U-Net | +0.013 | +0.020 | +0.045 |

This shows that bounding-box priors remain useful even after the base U-Net is trained with a better loss.

In the modified setting, the improvement from OralBBNet over U-Net is smaller than in the original setting because the modified U-Net is already strong. Nevertheless, OralBBNet still improves mean tooth/channel Dice from `0.826` to `0.871`. This is the most important difference for the assignment, because the final task is instance segmentation over 32 FDI channels rather than only binary tooth segmentation.

### 7.2 Efficiency and Training Cost

U-Net is computationally simpler. It uses only one image channel and has a smaller checkpoint:

```text
Modified U-Net checkpoint:      about 396 MB
Modified OralBBNet checkpoint:  about 435 MB
```

OralBBNet is therefore slightly heavier because it processes 32 extra prior channels and has an additional prior-gating branch. Its per-step computation and memory usage are expected to be higher than U-Net.

The efficiency trade-off can be summarized as follows:

| Aspect | U-Net | OralBBNet |
| --- | --- | --- |
| Input channels | 1 image channel | 32 prior channels + 1 image channel |
| Extra prior branch | No | Yes |
| Checkpoint size | Smaller, about 396 MB | Larger, about 435 MB |
| Expected training speed | Faster per step | Slower per step |
| Required annotations at inference | Image only | Image + bounding-box prior |
| Binary segmentation | Strong | Stronger |
| Tooth identity alignment | Weaker | Stronger |
| Practical limitation | No explicit tooth-location prior | Depends on prior-box quality |

However, OralBBNet is more efficient in terms of accuracy per fixed training budget. Under the same 60-epoch setting and threshold `0.5`, it gives higher Binary Dice, higher Binary IoU, and higher mean tooth/channel Dice. The spatial prior reduces the burden on the network: instead of learning tooth identity only from appearance, the model receives approximate tooth locations through the bounding-box maps.

The current logs do not record exact wall-clock time per epoch, so this report does not claim a precise training-time speedup. A fair timing comparison should add timestamp logging around each training call. Based on model structure, U-Net is faster and lighter; based on segmentation quality under the same epoch budget, OralBBNet is more effective.

### 7.3 Why OralBBNet Helps

The advantage of OralBBNet is most visible for tooth identity alignment. Many teeth have similar texture and shape, especially symmetric teeth or neighboring teeth. U-Net must infer the tooth number only from visual appearance and relative position. OralBBNet directly receives a spatial prior for each tooth channel, so the decoder is guided toward the correct anatomical region.

This explains why OralBBNet improves mean tooth/channel Dice more clearly than Binary Dice. Binary Dice only asks whether tooth pixels are found. Mean tooth/channel Dice also asks whether they are assigned to the correct FDI channel.

For this reason, OralBBNet is more suitable when the goal is a structured dental chart or a downstream disease analysis system that needs tooth identity. U-Net is attractive when inference must be simple and no detector or bounding-box prior is available.

## 8. Discussion

The original OralBBNet already outperforms the original U-Net, confirming that bounding-box priors are useful for this task. The modified OralBBNet further improves the result by correcting the loss direction and using a more suitable activation function.

The supplementary threshold-calibration experiment shows that a lower threshold can slightly increase Dice. However, the tuned thresholds can be as low as `0.05`, which indicates that the probability calibration is not ideal. For this reason, the main comparison in this report uses threshold `0.5`, which is simpler, more reproducible, and better aligned with the reference protocol.

For completeness, the tuned-threshold supplementary results are:

| Method | Tuned threshold | Binary Dice | Binary IoU | Mean tooth/channel Dice |
| --- | ---: | ---: | ---: | ---: |
| Modified U-Net tuned | 0.05 | 0.896 | 0.813 | 0.827 |
| Modified OralBBNet tuned | 0.05 | 0.909 | 0.834 | 0.871 |

The tuned threshold gives only a small improvement over fixed `0.5`. Therefore, using `0.5` is reasonable for the main report and keeps the comparison more conservative.

The main limitation is that the OralBBNet prior maps are built from dataset-provided YOLO labels or fallback boxes. A complete clinical pipeline should train a detector and evaluate the segmentation model using predicted boxes, not annotation-derived boxes. This would better measure real deployment performance.

## 9. Conclusion

At threshold `0.5`, the modified OralBBNet achieves the best overall result:

```text
Binary Dice:              0.907
Binary IoU:               0.831
Mean tooth/channel Dice:  0.871
```

Compared with the original OralBBNet, the modified OralBBNet improves Binary Dice by `0.044`, Binary IoU by `0.063`, and mean tooth/channel Dice by `0.115`. Compared with the modified U-Net, it still improves the main metrics, especially tooth/channel alignment.

Overall, OralBBNet is more accurate than U-Net because bounding-box priors provide explicit spatial guidance for each tooth position. U-Net remains lighter and simpler, but OralBBNet is the better choice when the goal is accurate 32-channel tooth instance segmentation.

## References

[1] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015.

[2] T. F. Chan and L. A. Vese. Active Contours Without Edges. *IEEE Transactions on Image Processing*, 10(2):266-277, 2001.

[3] D. Budagam et al. OralBBNet: Spatially Guided Dental Segmentation of Panoramic X-Rays with Bounding Box Priors. *arXiv preprint arXiv:2406.03747*, 2025.

[4] D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. In *International Conference on Learning Representations (ICLR)*, 2015.
