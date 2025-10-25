# 2D OASIS Brain Segmentation with an Improved UNet

**Author:** Muhammad Sakhran Thayyib / s49063581
**Task:** 1.3.1 (Easy Difficulty)

---

## 1) Problem Description

This project addresses 2D brain segmentation on the OASIS dataset (COMP3710 Task 1.3.1). Objective: train an **Improved UNet** to segment brain regions from 2D T1-weighted MRI slices. Required benchmark: **Dice Similarity Coefficient (DSC) ≥ 0.9** on the test set.

---

## 2) Algorithm and Implementation

### Model Architecture

Core model in `modules.py`: **Improved UNet** with the following enhancements:

* **Residual Blocks:** `ConvBlock` uses residual connections to improve gradient flow and enable deeper training.
* **Instance Normalization:** `InstanceNorm2d` normalizes per channel, per sample; stabilizes training for image-to-image tasks and small batch sizes.
* **Deep Architecture:** 5 down/upsampling levels with a **1024-channel bottleneck** for large receptive field and strong global context.

### Data Loading and Preprocessing

Handled by `dataset.py`:

* **Data Source:** `/home/groups/comp3710/OASIS`
* **Format:** Pre-split 2D PNGs

  * Images: `keras_png_slices_train/`, `keras_png_slices_validate/`, `keras_png_slices_test/`
  * Masks: e.g., `keras_png_slices_seg_train/` (matching splits for validation/test)
* **Loading:** OpenCV (`cv2`)
* **Preprocessing:**

  * Masks binarized (background=0, brain=1)
  * Images and masks resized to **256×256**
  * **Normalization:** map input intensities to **[-1, 1]** with mean=0.5, std=0.5

### Training

Orchestrated by `train.py`:

* **Loss:** `BCEWithLogitsLoss` + **DiceLoss** → `L_total = L_bce + L_dice`
* **Optimizer:** `AdamW`, `lr = 1e-4`
* **Mixed Precision:** `torch.cuda.amp`
* **Augmentation (Albumentations):**

  * Random H/V flips and 90° rotations
  * Random shift/scale/rotate
  * Elastic transforms
* **Validation & Checkpointing:** Evaluate each epoch on `keras_png_slices_validate/`; save best model to **`best_oasis_unet.pth`**. Logs in `oasis_unet_*.out`.

---

## 3) Dependencies

Conda env (Python 3.11).

```bash
# Activate env
source ~/miniconda3/bin/activate torch

# Ensure pip
conda install -y pip

# PyTorch CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Other deps
pip install scikit-learn scipy opencv-python-headless
pip install "albumentations==1.3.1" "albucore==0.0.9"

# nibabel not required (working with PNGs)
```

---

## 4) How I Ran the Code

### 1) Submitted the training job

I placed all `.py` scripts and `runner_unet` in the project directory and submitted:

```bash
sbatch runner_unet
```

I monitored the job:

```bash
squeue -u s49063581
```

The best validation checkpoint was saved as **`best_oasis_unet.pth`**.

### 2) Evaluated on the test set

After training completed and `best_oasis_unet.pth` existed, I evaluated:

```bash
# (torch) env active
python evaluate_test.py
# or via Slurm
sbatch runner_evaluate
```

This ran on `keras_png_slices_test/` and printed the average Dice.

### 3) Generated predictions (optional visualization)

I produced example masks with:

```bash
# (torch) env active
python predict.py
```

It loaded `best_oasis_unet.pth`, segmented samples from `keras_png_slices_test/`, and wrote binary PNGs to `predictions/`.


## 5) Results

Training ran **10 epochs** before cancellation due to cluster time limit.

**Validation (Epoch 10, best):**

| Metric              | Value  | Set        |
| ------------------- | ------ | ---------- |
| Best Dice Score     | 0.9926 | Validation |
| Avg Training Loss   | 0.0346 | Training   |
| Avg Validation Loss | 0.0193 | Validation |

**Test (using best checkpoint from Epoch 10):**

| Metric             | Value  | Set  |
| ------------------ | ------ | ---- |
| Average Dice Score | 0.9919 | Test |

**Discussion:** Requirement ≥0.9 DSC on test. Achieved **0.9919** after 10 epochs. The Improved UNet, BCE+Dice loss, and augmentations were effective. High validation (0.9926) and test (0.9919) indicate strong generalization. Additional training unnecessary for the assignment target.



