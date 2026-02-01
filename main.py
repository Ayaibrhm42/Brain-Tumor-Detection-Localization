#!/usr/bin/env python3

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
import json
from pathlib import Path
from skimage import segmentation, measure, filters
from scipy import ndimage
from sklearn.model_selection import KFold, train_test_split
from google.colab import drive
from ultralytics import YOLO


# SECTION 1: PREPROCESSING 
# Images were inconsistent in size, contrast, noise, and brightness.
# This section standardizes, normalizes, reduces noise, enhances tumor areas,
# and sharpens edges for clearer images.

def setup_paths():
    """Setup and mount Google Drive paths"""
    drive.mount('/content/drive')
    
    BASE_PATH = '/content/drive/MyDrive/BrainTumorData'
    IMAGES_PATH = os.path.join(BASE_PATH, 'images')
    LABELS_PATH = os.path.join(BASE_PATH, 'labels')
    
    MODERATE_PATH = '/content/moderate_preprocessed_brain_data'
    MODERATE_IMAGES_PATH = os.path.join(MODERATE_PATH, 'images')
    MODERATE_LABELS_PATH = os.path.join(MODERATE_PATH, 'labels')
    SEGMENTATION_PATH = os.path.join(MODERATE_PATH, 'segmentations')
    
    os.makedirs(MODERATE_IMAGES_PATH, exist_ok=True)
    os.makedirs(MODERATE_LABELS_PATH, exist_ok=True)
    os.makedirs(SEGMENTATION_PATH, exist_ok=True)
    
    return {
        'base': BASE_PATH,
        'images': IMAGES_PATH,
        'labels': LABELS_PATH,
        'moderate': MODERATE_PATH,
        'moderate_images': MODERATE_IMAGES_PATH,
        'moderate_labels': MODERATE_LABELS_PATH,
        'segmentation': SEGMENTATION_PATH
    }

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'target_size': (512, 512),
    'brightness_adjustment': 1.2,    # 20% brightness boost
    'contrast_enhance': True,
    'clip_limit': 1.5,              # Gentle CLAHE
    'tile_grid_size': (8, 8),
    'noise_reduction': True,
    'denoise_strength': 5,           # Light denoising
    'sharpening': True,
    'sharpen_strength': 0.2,         # Subtle sharpening
    'enable_tumor_enhancement': True
}

def moderate_preprocess_image(img_path, config):
    """Apply preprocessing with CLAHE, denoising, and sharpening"""
    try:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return None
        
        # Step 1: Brightness adjustment
        img_gray = np.clip(img_gray * config['brightness_adjustment'], 0, 255).astype(np.uint8)
        
        # Step 2: Contrast enhancement with CLAHE
        if config['contrast_enhance']:
            clahe = cv2.createCLAHE(clipLimit=config['clip_limit'],
                                   tileGridSize=config['tile_grid_size'])
            img_gray = clahe.apply(img_gray)
        
        # Step 3: Noise reduction
        if config['noise_reduction']:
            img_gray = cv2.fastNlMeansDenoising(img_gray, None,
                                               h=config['denoise_strength'],
                                               templateWindowSize=7,
                                               searchWindowSize=15)
        
        # Step 4: Sharpening
        if config['sharpening']:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]]) * config['sharpen_strength']
            kernel[1, 1] = 1 + (8 * config['sharpen_strength'])
            img_gray = cv2.filter2D(img_gray, -1, kernel)
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        
        # Step 5: Resize to target size
        img_resized = cv2.resize(img_gray, config['target_size'], 
                                interpolation=cv2.INTER_LANCZOS4)
        
        return img_resized
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

def create_tumor_contrast_enhancement(img_gray, label_path):
    """Enhance contrast specifically in tumor regions"""
    try:
        with open(label_path, 'r') as f:
            labels = f.readlines()
        
        if not labels:
            return img_gray
        
        h, w = img_gray.shape
        tumor_mask = np.zeros((h, w), dtype=np.uint8)
        
        for label in labels:
            parts = label.strip().split()
            if len(parts) >= 5:
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert normalized coords to pixel coords
                x_center *= w
                y_center *= h
                width *= w
                height *= h
                
                x1 = int(x_center - width/2)
                y1 = int(y_center - height/2)
                x2 = int(x_center + width/2)
                y2 = int(y_center + height/2)
                
                tumor_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
        
        # Apply stronger CLAHE to tumor regions
        clahe_tumor = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_tumor = clahe_tumor.apply(img_gray)
        
        # Blend enhanced regions with original
        tumor_mask_3ch = tumor_mask.astype(float) / 255.0
        result = img_gray.astype(float) * (1 - tumor_mask_3ch) + \
                 enhanced_tumor.astype(float) * tumor_mask_3ch
        
        return result.astype(np.uint8)
        
    except Exception as e:
        return img_gray

def preprocess_dataset(paths, config):
    """Preprocess entire dataset"""
    print("PREPROCESSING DATASET")
    
    image_files = sorted(glob.glob(os.path.join(paths['images'], '*.jpg')))
    print(f"Found {len(image_files)} images to process\n")
    
    processed_count = 0
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(paths['labels'], label_name)
        
        processed_img = moderate_preprocess_image(img_path, config)
        
        if processed_img is not None:
            if config['enable_tumor_enhancement'] and os.path.exists(label_path):
                processed_img = create_tumor_contrast_enhancement(processed_img, label_path)
            
            output_path = os.path.join(paths['moderate_images'], img_name)
            cv2.imwrite(output_path, processed_img)
            
            if os.path.exists(label_path):
                shutil.copy2(label_path, 
                           os.path.join(paths['moderate_labels'], label_name))
            
            processed_count += 1
    
    print(f"\n✓ Processed {processed_count}/{len(image_files)} images")
    
    # Backup to Google Drive
    drive_backup_path = '/content/drive/MyDrive/BrainTumorData_Moderate_Enhanced'
    if os.path.exists(drive_backup_path):
        shutil.rmtree(drive_backup_path)
    shutil.copytree(paths['moderate'], drive_backup_path)
    print(f"   Backup saved to: {drive_backup_path}")



# SECTION 2: K-FOLD CROSS-VALIDATION TRAINING

def save_splits(train_val_names, test_names, fold_splits, splits_file):
    """Save train/val/test splits to JSON file"""
    splits_data = {
        'train_val_images': train_val_names,
        'test_images': test_names,
        'folds': fold_splits,
        'n_folds': len(fold_splits),
        'test_split_ratio': 0.2
    }
    
    with open(splits_file, 'w') as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"\nSplits saved to: {splits_file}")

def load_splits(splits_file):
    """Load previously saved splits"""
    with open(splits_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded existing splits from: {splits_file}")
    print(f"   Train/Val images: {len(data['train_val_images'])}")
    print(f"   Test images: {len(data['test_images'])}")
    
    return data['train_val_images'], data['test_images'], data['folds']

def create_new_splits(all_image_names, n_folds=5, test_ratio=0.2):
    """Create new train/val/test splits with k-fold"""
    train_val_names, test_names = train_test_split(
        all_image_names, test_size=test_ratio, random_state=42
    )
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_splits = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_val_names)):
        train_fold = [train_val_names[i] for i in train_idx]
        val_fold = [train_val_names[i] for i in val_idx]
        
        fold_splits.append({
            'fold': fold_idx + 1,
            'train': train_fold,
            'val': val_fold
        })
    
    return train_val_names, test_names, fold_splits

def create_fold_dataset(data_path, fold_data, fold_num, kfold_base):
    """Create dataset structure for a specific fold"""
    fold_path = os.path.join(kfold_base, f'fold_{fold_num}')
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(fold_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(fold_path, 'labels', split), exist_ok=True)
    
    # Copy train images/labels
    for img_name in fold_data['train']:
        img_src = os.path.join(data_path, 'images', img_name)
        label_src = os.path.join(data_path, 'labels', img_name.replace('.jpg', '.txt'))
        
        shutil.copy2(img_src, os.path.join(fold_path, 'images', 'train', img_name))
        if os.path.exists(label_src):
            shutil.copy2(label_src, 
                        os.path.join(fold_path, 'labels', 'train', 
                                   img_name.replace('.jpg', '.txt')))
    
    # Copy val images/labels
    for img_name in fold_data['val']:
        img_src = os.path.join(data_path, 'images', img_name)
        label_src = os.path.join(data_path, 'labels', img_name.replace('.jpg', '.txt'))
        
        shutil.copy2(img_src, os.path.join(fold_path, 'images', 'val', img_name))
        if os.path.exists(label_src):
            shutil.copy2(label_src, 
                        os.path.join(fold_path, 'labels', 'val', 
                                   img_name.replace('.jpg', '.txt')))
    
    # Create data.yaml
    yaml_content = f"""path: {fold_path}
train: images/train
val: images/val

nc: 1
names: ['tumor']
"""
    
    with open(os.path.join(fold_path, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    
    return os.path.join(fold_path, 'data.yaml')

def train_fold(fold_num, data_yaml, results_dir, epochs=150):
    """Train YOLOv8 model on a specific fold"""
    print(f"TRAINING FOLD {fold_num}")
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=512,
        batch=16,
        patience=20,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        project=results_dir,
        name=f'fold_{fold_num}',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        deterministic=True,
        mosaic=0.0,
        close_mosaic=120,
        seed=0
    )
    
    val_results = model.val()
    
    print(f"\n   FOLD {fold_num} RESULTS:")
    print(f"   mAP50: {val_results.box.map50:.3f}")
    print(f"   mAP50-95: {val_results.box.map:.3f}")
    print(f"   Precision: {val_results.box.mp:.3f}")
    print(f"   Recall: {val_results.box.mr:.3f}")
    
    return val_results

def run_kfold_training(n_folds=5, epochs=150):
    """Run complete k-fold cross-validation training"""
    drive.mount('/content/drive')
    
    DATA_PATH = '/content/drive/MyDrive/BrainTumorData_Moderate_Enhanced'
    SPLITS_FILE = '/content/drive/MyDrive/brain_tumor_splits.json'
    KFOLD_BASE = '/content/kfold_training'
    RESULTS_DIR = os.path.join(KFOLD_BASE, 'results')
    
    all_images = [f for f in os.listdir(os.path.join(DATA_PATH, 'images')) 
                  if f.endswith('.jpg')]
    
    # Load or create splits
    if os.path.exists(SPLITS_FILE):
        train_val_names, test_names, fold_splits = load_splits(SPLITS_FILE)
    else:
        train_val_names, test_names, fold_splits = create_new_splits(
            all_images, n_folds=n_folds
        )
        save_splits(train_val_names, test_names, fold_splits, SPLITS_FILE)
    
    # Train each fold
    fold_results = []
    
    for fold_data in fold_splits:
        fold_num = fold_data['fold']
        data_yaml = create_fold_dataset(DATA_PATH, fold_data, fold_num, KFOLD_BASE)
        val_results = train_fold(fold_num, data_yaml, RESULTS_DIR, epochs=epochs)
        
        fold_results.append({
            'fold': fold_num,
            'map50': val_results.box.map50,
            'map50_95': val_results.box.map,
            'precision': val_results.box.mp,
            'recall': val_results.box.mr
        })
    
    # Print summary
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    
    avg_map50 = np.mean([r['map50'] for r in fold_results])
    avg_map50_95 = np.mean([r['map50_95'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_recall = np.mean([r['recall'] for r in fold_results])
    
    print(f"\nAverage Performance Across {n_folds} Folds:")
    print(f"   mAP50: {avg_map50:.3f}")
    print(f"   mAP50-95: {avg_map50_95:.3f}")
    print(f"   Precision: {avg_precision:.3f}")
    print(f"   Recall: {avg_recall:.3f}")
    
    return fold_results


# MAIN EXECUTION

def main():
    """Main execution pipeline"""
    print("BRAIN TUMOR DETECTION PIPELINE")
    
    # Step 1: Preprocessing
    print("Step 1: Preprocessing data...")
    paths = setup_paths()
    preprocess_dataset(paths, PREPROCESSING_CONFIG)
    
    # Step 2: K-Fold Training
    print("\n\nStep 2: Running k-fold cross-validation training...")
    fold_results = run_kfold_training(n_folds=5, epochs=150)
    
    print("\nPipeline completed successfully!")
    
    return fold_results


if __name__ == "__main__":
    results = main()
