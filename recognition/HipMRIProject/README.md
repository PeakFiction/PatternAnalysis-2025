# 2D HipMRI Prostate Segmentation with an Improved UNet

**Author:** Muhammad Sakhran Thayyib, 4906358 / s4906358  
**Task:** 1.3.3 (Normal Difficulty)

---

## 1) Problem Description

2D prostate segmentation on the **HipMRI Study on Prostate Cancer** dataset (COMP3710 Task 1.3.3) using an **Improved UNet** to segment multiple organs from 2D T1-weighted MRI slices.

### Classes
| Label | Class |
|--------|--------|
| 0 | Background |
| 1 | Body |
| 2 | Bone |
| 3 | Bladder |
| 4 | Rectum |
| 5 | Prostate |

**Benchmark:** Dice Similarity Coefficient (DSC) ≥ **0.75** on the test set for **Prostate (Class 5)**.

---

## 2) Algorithm and Implementation

### Model Architecture
Defined in `modules.py`, an **Improved UNet** (based on *nnU-Net* principles) with:

- **DoubleConv Blocks:** Each encoder/decoder stage uses two sequential `(Conv2d → InstanceNorm2d → LeakyReLU)` operations.  
- **InstanceNorm2d:** Used instead of `BatchNorm2d` to normalize each image individually, improving consistency for medical images with varying contrast.  
- **LeakyReLU:** Used instead of ReLU for more stable gradients.  
- **Deep Architecture:** 5 down/upsampling levels with a 1024-channel bottleneck.  
- **Upsampling:** Decoder uses bilinear upsampling (`mode="bilinear"`) for smooth non-learned scaling.

---

### Data Loading and Preprocessing
Managed by `dataset.py`.

**Data Source:**  
`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/`

**Format:** Pre-split 2D NIfTI (`.nii.gz`)

**Directories:**
```

keras_slices_train/
keras_slices_validate/
keras_slices_test/
keras_slices_seg_train/
keras_slices_seg_validate/
keras_slices_seg_test/

````

**Loading:** `nibabel`

**Preprocessing:**
- **Normalization:** Images scaled to `[0, 1]`  
- **Resizing:** All images/masks → `(256, 128)` using `F.interpolate`  
  - Images: `mode="bilinear"`  
  - Masks: `mode="nearest"` (preserves integer class labels)

---

### Training
Orchestrated by `train.py` and `test.py`.

| Component | Configuration |
|------------|----------------|
| **Loss** | `CrossEntropyLoss` for multi-class segmentation |
| **Optimizer** | `Adam`, `lr=1e-4` |
| **Validation & Checkpointing** | Evaluate on `keras_slices_validate/` each epoch; save highest prostate Dice model to `best_hipmri_unet.pth` |
| **Logging** | Slurm stdout captured in `hipmri_unet_*.out`, `hipmri_test_*.out` |

---

## 3) Dependencies

Conda environment (Python 3.11) on **Rangpur**.

```bash
# Activate env
source ~/miniconda3/bin/activate torch

# Ensure pip
conda install -y pip

# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Other dependencies
pip install nibabel
pip install matplotlib
pip install numpy
````

---

## 4) How I Ran the Code

### 1) Submitted the Training Job

Placed all `.py` scripts and `runner_hipmri` in the project directory, made the runner executable, and submitted:

```bash
chmod +x runner_hipmri
sbatch runner_hipmri
```

Monitored progress with:

```bash
squeue -u s4906358
```

The job hit the cluster’s **20-minute limit** after ~20 epochs (`hipmri_unet_322320.out`), but the best validation checkpoint (from **Epoch 3**) was already saved as `best_hipmri_unet.pth`.

---

### 2) Evaluated on the Test Set

Created `test.py` and a `runner_test` Slurm file, then ran:

```bash
chmod +x runner_test
sbatch runner_test
```

This loaded `best_hipmri_unet.pth`, evaluated on `keras_slices_test/`, and printed class-wise Dice (`hipmri_test_322321.out`).

---

### 3) Generated Predictions (Optional Visualization)

Produced example masks with:

```bash
# (torch) env active
python predict.py
```

This loaded `best_hipmri_unet.pth`, segmented a sample from `keras_slices_test/`, and saved `prediction_visualization.png`.

---

## 5) Prediction Result

---<img width="1405" height="504" alt="prediction_visualization" src="https://github.com/user-attachments/assets/68e58dda-367e-488a-9d2b-2b4645dce74d" />


## 6) Results

**Training:** ~20 epochs before timeout (20-minute limit).
**Validation (Best, from Epoch 3):**

| Metric                 | Value  | Set        |
| ---------------------- | ------ | ---------- |
| **Best Prostate Dice** | 0.8133 | Validation |

**Test (Best checkpoint from Epoch 3):**

| Class | Label      | Dice Score |
| ----- | ---------- | ---------- |
| 0     | Background | 0.9970     |
| 1     | Body       | 0.9847     |
| 2     | Bone       | 0.9245     |
| 3     | Bladder    | 0.9518     |
| 4     | Rectum     | 0.8696     |
| 5     | Prostate   | 0.8400     |

---

## Discussion

**Requirement:** Prostate Dice ≥ 0.75 on test
**Achieved:** 0.8400 ✅

The **Improved UNet**, using `InstanceNorm2d` and `LeakyReLU`, proved highly effective.
The best model emerged very early (**Epoch 3**) and generalized extremely well to the test set, exceeding the target benchmark.
