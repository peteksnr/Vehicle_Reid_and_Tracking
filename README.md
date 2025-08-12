# 🚗 Vehicle Re-identification and Tracking System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

> A deep learning-based system for vehicle re-identification and tracking using the **OSNet** architecture and **DeepSORT** algorithm. Includes dataset preparation, model training, evaluation, and video processing.

---

## 📦 Dependencies

This project requires the following Python packages:

- `torch` (PyTorch)
- `torchvision`
- `tqdm`
- `numpy`
- `opencv-python` (`cv2`)
- `gdown`
- `ultralytics`
- `scikit-learn`

---

## ⚙️ Installation

### 1️⃣ Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 2️⃣ Install the dependencies:
```bash
pip install torch torchvision tqdm numpy opencv-python gdown ultralytics scikit-learn
```
---

## 📁 Project Structure
```bash
organize.py             # Organizes the VeRi dataset into train/test/query
train_reid_model.py     # Train the vehicle re-identification model
evaluate_reid.py        # Evaluate the trained re-identification model
video_evaluation.py     # Process video for vehicle tracking and identification
osnet.py                # OSNet architecture implementation
deep_sort_tracker.py    # DeepSORT tracking implementation
model.pth               # Pre-trained model weights
VeRi/                   # Raw VeRi dataset
veri_data/              # Organized dataset for training/evaluation
```
## 🚀 Usage
### 1️⃣ Data Preparation
```bash
python organize.py
```   
This will:   
	•	Create train/, test/, and query/ directories   
	•	Organize images according to vehicle IDs   
	•	Prepare the dataset for training and evaluation   
 
### 2️⃣ Train the Model
```bash
python train_reid_model.py
```
Trains the OSNet-based re-identification model using the data in veri_data/train.

### 3️⃣ Evaluate the Model
```bash
python evaluate_reid.py
```
Evaluates the model’s performance on the veri_data/test set.

### 4️⃣ Process Video for Tracking
```bash
python video_evaluation.py
```
Runs vehicle tracking and identification on video files.

---

## 🚀 Usage
```bash
veri_data/
├── train/
├── test/
└── query/
```   
Before running training or evaluation:   
	1.	Place your VeRi dataset in the data/ directory   
	2.	Run organize.py to structure it properly    
	3.	The script will populate veri_data/   
---
## 🧠 Model Architecture
	•	OSNet backbone for re-identification
	•	Multi-task learning for:
	•	Vehicle ID classification
	•	Color classification
	•	Vehicle type classification
	•	Batch-hard triplet loss for discriminative embeddings
	•	Adaptive triplet loss for stability
	•	DeepSORT for multi-object tracking in videos

---
## 📌 Notes
	•	GPU acceleration is used if available (falls back to CPU otherwise)
	•	Training progress and metrics are displayed in the terminal
	•	Model checkpoints are saved automatically
	•	Always run organize.py before training or evaluation

---

## 🤝 Acknowledgments
	•	VeRi Dataset for vehicle re-identification research
	•	PyTorch for the deep learning framework
	•	DeepSORT for object tracking



