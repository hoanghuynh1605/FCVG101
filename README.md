# Rice Leaf Disease Diagnosis

---

## 1. Project Overview & Scenario
This project delivers a production-ready Minimum Viable Product (MVP) computer vision inspection assistant. The system aids farm managers, agronomists, and warehouse supervisors in detecting, localizing, and mitigating crop foliar diseases across three key pathologies:
1. **Bacterial Leaf Blight** (*Xanthomonas oryzae*)
2. **Brown Spot** (*Bipolaris oryzae*)
3. **Leaf Smut** (*Entyloma oryzae*)

The codebase consists of:
- **Core Inspection Pipeline** (`pipeline.ipynb` & `app.py`): An end-to-end automated workflow from raw image conditioning, feature extraction, and transfer-learning deep learning to Grad-CAM lesion explainability.
- **Camera Calibration Module** (`calibration.ipynb`): An camera geometry and lens undistortion engine addressing Requirement (a) of the coursework specification.

---

## 2. System Architecture & Engineering Modules

### Core Automated Disease Inspection Pipeline (`pipeline.ipynb` & `app.py`)
```
Original Image Dataset (120 field images across 3 classes)
      │
      ▼
TASK 1: Classical Computer Vision Engine
      │ ├── Linear & Non-Linear Spatial Filters (Box, Gaussian, Median, Bilateral with PSNR/MSE)
      │ ├── Gradient & Edge Operators (Sobel X/Y, Gradient Magnitude, Laplacian, Canny Hysteresis)
      │ ├── 2D FFT Frequency Analysis (Fourier Magnitude Spectrum, Circular Low-Pass & High-Pass)
      │ └── Interest Point Extraction & Matching (Harris Corners, SIFT 128-D, ORB, Two-View SIFT Matching)
      ▼
Processed Image Dataset (`processed_rice_leaf_diseases/`)
      │ (Bilateral edge-preserving denoising + CIE LAB CLAHE contrast normalization)
      ▼
Standardized Geometric Resize (224 × 224 with `cv2.INTER_AREA`)
      │
      ▼
TASK 2: Deep Learning Classification Pipeline & Model Refinement
      │ ├── Pixel Normalization: `layers.Rescaling(1./255)` embedded into model
      │ ├── Data Augmentation: `RandomFlip`, `RandomRotation(0.2)`, `RandomZoom(0.1)`
      │ ├── Stratified Train/Val Split: 80% Train (96) / 20% Val (24)
      │ ├── Pretrained MobileNetV2 Backbone + GAP + Dropout(0.3) + Dense(128) + Dropout(0.2) + Dense(3)
      │ └── Rigorous Evaluation: 91.67% Validation Accuracy, Confusion Matrix, R² = 0.8532, MAE = 0.0833
      ▼
TASK 3: Streamlit Assistant Feature & Application Integration (`app.py`)
      │ ├── Intent Recognition & Modular Tool Routing (6 Interactive Modes)
      │ ├── End-to-End Visual Inspection Pipeline (Preproc → DL Inference → Diagnosis)
      │ ├── Grad-CAM Explainability (Gradient attribution attention heatmaps on leaf lesions)
```

---

## 3. Deliverables & Project Directory Structure

```
fcvg101/
├── calibration.ipynb                  # Camera Calibration & Undistortion Notebook
├── calibration_images/                # 13 checkerboard calibration frames
├── app.py                             # Streamlit Assistant Web Application (Task 3)
├── pipeline.ipynb                     # Complete executed Task 1 & Task 2 Jupyter Notebook
├── README.md                          # Project documentation and engineering manual
├── class_names.json                   # Class mapping configuration
├── models/
│   └── rice_leaf_inspection_model.keras  # Saved refined CNN model (10.49 MB, 91.67% val acc)
├── outputs/
│   ├── calibration_undistortion.png   # Calibration corners & undistortion comparison
│   ├── task2_confusion_matrix.png     # Validation confusion matrix heatmap
│   ├── task2_training_curves.png      # Training & validation accuracy/loss trajectories
│   ├── task2_dataset_samples.png      # Visual sample gallery across the 3 classes
│   └── task2_metrics.json             # Quantitative performance metrics
├── processed_rice_leaf_diseases/      # Processed dataset generated via Task 1
│   ├── Bacterial leaf blight/         # 40 bilateral + CLAHE enhanced images
│   ├── Brown spot/                    # 40 bilateral + CLAHE enhanced images
│   └── Leaf smut/                     # 40 bilateral + CLAHE enhanced images
├── rice_leaf_diseases/                # Raw 120-image rice leaf pathology dataset
└── test_samples/                      # Independent test evaluation samples
```

---

## 4. Assistant Capabilities & Intent Routing in Streamlit (`app.py`)

The Streamlit web application provides **6 specialized assistant tools**:

1. **Automated Inspection Assistant (End-to-End)**:
   - Ingests raw leaf image from user upload, test samples, or dataset explorer.
   - Runs classical Task 1 bilateral denoising + LAB CLAHE enhancement.
   - Scales to 224x224 and passes through the refined MobileNetV2 DL model.
   - Computes **Grad-CAM Attention Heatmap** overlaid on the lesion spots.
   - Outputs disease diagnosis, confidence meter.

2. **Classical CV: Spatial Filtering & Denoising**:
   - Interactive Box, Gaussian, Median, and Bilateral edge-preserving filtering with live kernel size and sigma sliders.
   - Quantitative Salt-and-Pepper denoising benchmark computing PSNR (dB) and MSE.

3. **Classical CV: Edge Detection & Gradients**:
   - Computes directional Sobel derivatives, gradient magnitude, and Laplacian.
   - Dual-threshold Canny edge detector with interactive sliders and visual overlay.

4. **Classical CV: 2D FFT Frequency Analysis**:
   - 2D Discrete Fourier Transform with centered logarithmic power spectrum.
   - Interactive circular Low-Pass Filter (LPF) and High-Pass Filter (HPF) spatial reconstruction with dynamic radius cutoff slider.

5. **Classical CV: Feature Detection & Matching**:
   - Tri-detector interest point comparison: Harris Corners, SIFT 128-D scale/orientation keypoints, and ORB binary keypoints.
   - Two-View SIFT correspondence matching across simulated viewpoint rotations with Lowe's ratio test.

6. **Model Performance & System Evaluation**:
   - Complete quantitative audit: Validation Accuracy (**91.67%**), Discrete MAE (**0.0833**), Discrete R-Squared (**0.8532**), Probability MAE (**0.0981**), Probability R-Squared (**0.7957**).
   - High-resolution confusion matrix and 20-epoch training curves.

---

## 5. Quickstart: How to Run the Application & Modules

### Prerequisites
- Python 3.10+
- Required packages: `streamlit`, `opencv-python`, `tensorflow`, `numpy`, `matplotlib`, `seaborn`, `pillow`

### Running the Camera Calibration
Open and execute the notebook:
```bash
jupyter notebook calibration.ipynb
```
Outputs and undistortion figures are automatically saved to `outputs/calibration_undistortion.png`.

### Launching the Streamlit Assistant
Run the following command in the `fcvg101/` directory:
```bash
streamlit run app.py
```
