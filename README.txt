Vehicle Re-identification and Tracking System
===========================================

This project implements a vehicle re-identification and tracking system using deep learning. It includes training, evaluation, and video processing capabilities for vehicle tracking and identification.

Dependencies
-----------
The project requires the following Python packages:
- torch (PyTorch)
- torchvision
- tqdm
- numpy
- opencv-python (cv2)
- gdown
- ultralytics
- scikit-learn

Installation
-----------
1. Create a new Python virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```
   pip install torch torchvision tqdm numpy opencv-python gdown ultralytics scikit-learn
   ```

Project Structure
---------------
- organize.py: Script for organizing the VeRi dataset
- train_reid_model.py: Main script for training the re-identification model
- evaluate_reid.py: Script for evaluating the re-identification model
- video_evaluation.py: Script for processing video files
- osnet.py: Implementation of the OSNet architecture
- deep_sort_tracker.py: Implementation of the DeepSORT tracking algorithm
- model.pth: Pre-trained model weights
- VeRi/: Directory containing training and evaluation data
- veri_data/: Directory containing the VeRi dataset

Usage
-----
1. Data Preparation:
   ```
   python organize.py
   ```
   This script organizes the VeRi dataset into the required structure:
   - Creates train/, test/, and query/ directories
   - Organizes images according to vehicle IDs
   - Prepares the dataset for training and evaluation

2. Training the Model:
   ```
   python train_reid_model.py
   ```
   This will train the re-identification model using the data in the veri_data/train directory.

3. Evaluating the Model:
   ```
   python evaluate_reid.py
   ```
   This will evaluate the model's performance on the test set.

4. Processing Videos:
   ```
   python video_evaluation.py
   ```
   This will process video files for vehicle tracking and identification.

Data Organization
---------------
The project expects the following data structure:
- veri_data/
  - train/
  - test/
  - query/

Before running any training or evaluation scripts:
1. Place your VeRi dataset in the data/ directory
2. Run organize.py to properly structure the dataset
3. The script will create the necessary directory structure in veri_data/

Model Architecture
----------------
The system uses an OSNet-based architecture with the following features:
- Multi-task learning for ID, color, and vehicle type classification
- Batch-hard triplet loss for improved feature learning
- Adaptive triplet loss for better training stability

Notes
-----
- The model uses GPU acceleration if available, otherwise falls back to CPU
- Training progress and metrics are displayed during execution
- Model checkpoints are saved automatically during training
- Make sure to run organize.py before training or evaluation

For any issues or questions, please refer to the code comments or create an issue in the repository. 