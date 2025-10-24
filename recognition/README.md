Project: 2D OASIS Brain Segmentation with an Improved UNet

Author: (Your Name / sID)
Task: 1.3.1 (Easy Difficulty)

1. Problem Description

This project aims to solve the 2D brain segmentation task for the OASIS dataset, as specified in the COMP3710 Pattern Analysis assignment. The goal is to create a model based on an "Improved UNet" architecture that can accurately segment the brain from T1-weighted MRI scans. The target metric is a minimum Dice Similarity Coefficient (DSC) of 0.9 on the test set.

2. Algorithm and Implementation

Model Architecture

The model implemented in modules.py is an Improved UNet. The key improvements over a standard UNet are:

Residual Blocks: Each convolutional block (ConvBlock) includes a residual (skip) connection. This helps with gradient flow and allows for a deeper network to be trained effectively.

Instance Normalization: InstanceNorm2d is used instead of BatchNorm2d. Instance Norm is common in image-to-image tasks (like segmentation and style transfer) as it normalizes features per-channel, per-sample, which is beneficial when batch statistics are not representative (e.g., small batch sizes).

Deep Architecture: The model has 5 levels, downsampling to a 1024-channel bottleneck, providing a large receptive field to capture global context.

Data Loading and Preprocessing

The dataset.py script handles loading and preparing the data:

Path: It reads data from the shared Rangpur directory: /home/groups/comp3710/OASIS.

Loading: It uses nibabel to load .nii.gz files.

Preprocessing: Labels are binarized (all non-zero values are mapped to 1). All images and masks are resized to a uniform 256x256.

Normalization: Images are normalized to a [-1, 1] range (mean=0.5, std=0.5).

Training

Training is defined in train.py:

Loss Function: A combined loss of Binary Cross-Entropy (BCE) with Logits and Dice Loss is used. L_total = L_bce + L_dice. This combination provides pixel-level stability (from BCE) and addresses class imbalance (from Dice), which is standard for high-performance segmentation.

Optimizer: AdamW is used with a learning rate of 1e-4.

Augmentation: To achieve the 0.9 Dice target, heavy data augmentation is critical. The albumentations library is used for:

Random Flips & Rotations

Shift, Scale, Rotate

Elastic Transformations

Validation: The data is split into 80% training and 20% validation. The model checkpoint (best_oasis_unet.pth) is saved based on the highest validation Dice score.

3. Dependencies

To run this project, the following packages must be installed in your Conda environment:

# Activate your environment first: conda activate torch
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install nibabel
pip install scikit-learn
pip install albumentations
pip install scipy


4. How to Run the Code

1. Submit Training Job

To train the model, submit the job to the Slurm scheduler:

sbatch runner_unet


This will start the training process on an A100 GPU. You can monitor its progress and see the output in the oasis_unet_*.out file.

2. Run Prediction

Once training is complete and you have a best_oasis_unet.pth file, you can run prediction on a few test images:

python predict.py


This will load the trained model, segment the first 5 images from the dataset, and save the results in the predictions/ directory.

5. Results

(You must fill this section in after your model has trained)

After 100 epochs, the model achieved:

Final Training Loss: (Fill in)

Best Validation Dice Score: (Fill in)

(Include any plots of your loss/metrics here. You can add them to the README after downloading the .out file or using matplotlib to save plots during training).

(Briefly discuss if you met the 0.9 Dice target. If not, what could be improved? e.g., more epochs, different hyperparameters, Tversky loss, etc.)