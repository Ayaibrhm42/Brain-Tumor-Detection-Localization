# Brain Tumor Detection and Localization using YOLOv8

Automated detection and localization of brain tumors in MRI scans using YOLOv8 object detection with custom preprocessing pipeline.

## Project Overview

This project implements a computer vision pipeline for detecting and localizing brain tumors in MRI images. The model achieves **92.7% average mAP50** through careful preprocessing and YOLOv8 architecture optimization.

## Technical Approach

### Preprocessing Pipeline
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
- **Bilateral filtering** for noise reduction while preserving edges  
- **Image sharpening** to enhance tumor boundaries
- **Tumor-specific contrast enhancement** to improve detection
- **Brightness normalization** across dataset

### Model Architecture
- **YOLOv8n** (nano variant) for efficient inference
- **5-fold cross-validation** for robust performance evaluation
- Experimented with different preprocessing parameters to optimize detection

### Training Configuration
- **150 epochs** with early stopping (patience=20)
- **AdamW optimizer** with learning rate scheduling
- **Image size:** 512x512
- **Batch size:** 16
- Data augmentation disabled to focus on preprocessing impact

## Results

### Cross-Validation Performance

| Fold | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| 1 | 85.0% | 58.5% | 85.3% | 79.3% |
| 2 | 93.6% | 59.8% | 95.6% | 85.4% |
| 3 | 95.9% | 64.7% | 93.3% | 92.3% |
| 4 | 92.7% | 58.0% | 91.6% | 88.0% |
| 5 | **96.9%** | **66.3%** | 97.4% | 91.1% |
| **Average** | **92.7%** | **61.5%** | **92.6%** | **87.2%** |

**Key Findings:**
- Consistent high mAP50 (>85%) across all folds indicates robust model
- mAP50-95 lower than mAP50 suggests room for improvement in precise localization (IoU threshold sensitivity)
- High precision (92.6%) shows model is conservative in predictions
- Strong recall (87.2%) indicates good tumor detection coverage

## Technologies

- **YOLOv8** (Ultralytics) - Object detection framework
- **OpenCV** - Image preprocessing  
- **Python** 3.12
- **PyTorch** - Deep learning backend
- **Google Colab** - Training environment (T4 GPU)
- **scikit-learn** - K-fold splitting

## Key Learnings

- **Preprocessing is critical:** CLAHE with gentle clip limit (1.5) works best for MRI scans to enhance contrast without over-amplifying noise
- **Tumor-specific enhancement:** Targeted contrast enhancement on tumor regions improved model confidence
- **Model size vs performance:** YOLOv8n provides good balance between speed and accuracy for this application
- **Cross-validation consistency:** Low variance across folds (±4% mAP50) indicates stable preprocessing pipeline

## Limitations

- Test set evaluation incomplete in current version
