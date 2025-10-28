# 2D HipMRI Prostate Segmentation with an Improved UNet

**Author:** s4906358
**Task:** 1.3.3 (Normal Difficulty)

---

## 1) Problem Description

2D prostate segmentation on the **HipMRI Study on Prostate Cancer** dataset (COMP3710 Task 1.3.3) using an **Improved UNet** to segment multiple organs from 2D T1-weighted MRI slices.
**Classes:**

* 0: Background
* 1: Body
* 2: Bone
* 3: Bladder
* 4: Rectum
* 5: Prostate

**Benchmark:** Dice Similarity Coefficient (DSC) **≥ 0.75** on the test set for **Prostate (Class 5)**.

---

## 2) Algorithm and Implementation

### Model Architecture

Defined in `modules.py`, an Improved UNet with:

* **DoubleConv Blocks:** Each encoder/decoder stage uses two sequential `(Conv2d → BatchNorm2d → ReLU)` ops for stability and stronger feature learning.
* **BatchNorm2d:** After every convolution (before ReLU) to normalize activations and accelerate convergence.
* **Deep Architecture:** 5 down/upsampling levels with a **1024-channel bottleneck** for large receptive field.
* **Upsampling:** Decoder uses bilinear upsampling (`mode="bilinear"`) for smooth non-learned scaling.

### Data Loading and Preprocessing

Managed by `dataset.py`:

* **Data Source:** `/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/`
* **Format:** Pre-split **2D NIfTI** (`.nii.gz`)

  * **Images:** `keras_slices_train/`, `keras_slices_validate/`, `keras_slices_test/`
  * **Masks:** `keras_slices_seg_train/`, `keras_slices_seg_validate/`, `keras_slices_seg_test/`
* **Loading:** `nibabel`
* **Preprocessing:**

  * **Normalization:** images scaled to **[0, 1]**
  * **Resizing:** all images/masks → **(256, 128)** via `F.interpolate`

    * Images: `mode="bilinear"`
    * Masks: `mode="nearest"` (preserve integer class labels)

### Training

Orchestrated by `train.py` and `test.py`:

* **Loss:** `CrossEntropyLoss` for multi-class segmentation
* **Optimizer:** `Adam`, `lr=1e-4`
* **Validation & Checkpointing:** evaluate on `keras_slices_validate/` each epoch; save highest **prostate Dice** model to **`best_hipmri_unet.pth`**
* **Logging:** Slurm stdout captured in `hipmri_unet_*.out`, `hipmri_test_*.out`

---

## 3) Dependencies

Conda env (Python 3.11) on Rangpur.

```bash
# Activate env
source ~/miniconda3/bin/activate torch

# Ensure pip
conda install -y pip

# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Other dependencies
pip install nibabel
pip install matplotlib
pip install numpy
```

---

## 4) How I Ran the Code

### 1) Submitted the training job

I placed all `.py` scripts and `runner_hipmri` in the project directory, made the runner executable, and submitted:

```bash
chmod +x runner_hipmri
sbatch runner_hipmri
```

I monitored with:

```bash
squeue -u s4906358
```

The job hit the cluster’s 20-minute limit after ~21 epochs, but the best validation checkpoint had already been saved as **`best_hipmri_unet.pth`**.

### 2) Evaluated on the test set

Training was killed before final test, so I created `test.py` and a `runner_test` Slurm file, then ran:

```bash
chmod +x runner_test
sbatch runner_test
```

This loaded **`best_hipmri_unet.pth`**, evaluated on `keras_slices_test/`, and printed class-wise Dice.

### 3) Generated predictions (optional visualization)

I produced example masks with:

```bash
# (torch) env active
python predict.py
```

It loaded **`best_hipmri_unet.pth`**, segmented a sample from `keras_slices_test/`, and wrote a comparison image to `predictions/`.

---

Prediction Result:
<img width="1405" height="504" alt="prediction_visualization" src="https://github.com/user-attachments/assets/b55d1a39-e89a-4fc0-9d13-87078ef0b24c" />


## 5) Results

**Training:** ~21 epochs before timeout (20-minute limit).
**Validation (Best, from Epoch 11):**

| Metric             | Value  | Set        |
| ------------------ | ------ | ---------- |
| Best Prostate Dice | 0.8171 | Validation |

**Test (Best checkpoint from Epoch 11):**

| Class | Label      | Dice Score |
| ----: | ---------- | ---------: |
|     0 | Background |     0.9971 |
|     1 | Body       |     0.9849 |
|     2 | Bone       |     0.9296 |
|     3 | Bladder    |     0.9355 |
|     4 | Rectum     |     0.8626 |
|     5 | Prostate   | **0.8362** |

**Discussion:** Requirement: prostate Dice ≥ 0.75 on test. Achieved **0.8362**. The Improved UNet with `BatchNorm2d` and `CrossEntropyLoss` was effective. Best model emerged early (Epoch 11) and generalized well.

