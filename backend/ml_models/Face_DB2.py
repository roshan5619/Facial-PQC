import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import torch
import torch.nn as nn
import pennylane as qml
import seaborn as sns
from typing import List, Tuple
import shutil
import sys
sys.stdout.reconfigure(encoding='utf-8')




BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # Get current script directory path instead of changing working directory all the time

# Original dataset directories
FACE_DIR = os.path.join(BASE_DIR, "Dataset25k", "faces")        # you can change it as per your need
NONFACE_DIR = os.path.join(BASE_DIR, "Dataset25k", "nonfaces")   # you can change it as per your need

# Enhanced dataset directories
ENHANCED_FACE_DIR = os.path.join(BASE_DIR, "Dataset25k", "enhanced_faces")  # you can change it as per your need
ENHANCED_NONFACE_DIR = os.path.join(BASE_DIR, "Dataset25k", "enhanced_nonfaces") # you can change it as per your need

# ROI dataset (rebuilt every run)
ROI_FACE_DIR = os.path.join(BASE_DIR, "Dataset25k", "ROI_faces") # you can change it as per your need
ROI_NONFACE_DIR = os.path.join(BASE_DIR, "Dataset25k", "ROI_nonfaces") # you can change it as per your need

# Detection parameters only
N_QUBITS = 8  # KEEP 8 QUBITS FOR DETECTION
ROI_SIZE = (128, 128)
VQC_EPOCHS = 200
MAX_IMAGES = 25000 # This is for non-face images
MAX_IMAGES_FACES = 30000  # This is for face images ,Because in 30k images,there will be some invalid images.In next step we will filter them out and use only 25k good images for training.
# Enhancement parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAMMA_CORRECTION = 1.2


# 1. Load Haar Cascade
# I have downloaded and save the haarcascade_frontalface_default.xml in the working directory. You can also use the one provided by OpenCV.
haar_paths = [
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
    os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"),
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
]

face_cascade = None
for path in haar_paths:
    if os.path.exists(path):
        face_cascade = cv2.CascadeClassifier(path)
        if not face_cascade.empty():
            print(f"Loaded Haar cascade from: {path}")
            break

if face_cascade is None or face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade. Please ensure the file exists in one of the expected paths.")


# 2. Image Enhancement Functions

def enhance_image(image):
    """
    Apply enhancement pipeline to normalize image style
    Steps: Grayscale -> CLAHE -> Gamma Correction
    """
    # Convert to grayscale 
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(gray)
    
    # Apply gamma correction for brightness normalization
    enhanced = gamma_correction(enhanced, gamma=GAMMA_CORRECTION)
    
    return enhanced

def gamma_correction(image, gamma=1.0):
    """
    Applying gamma correction to normalize brightness
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction
    return cv2.LUT(image, table)

# ========================
# 3. Enhanced Dataset Preparation
# ========================
def prepare_enhanced_dataset(face_dir, nonface_dir, enhanced_face_dir, enhanced_nonface_dir):
    """
    Create enhanced versions of all images with consistent style
    """
    # Clear and recreate enhanced directories whenever this function is called
    for directory in [enhanced_face_dir, enhanced_nonface_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
    
    face_count, nonface_count = 0, 0

    # Preprocess face images
    if os.path.exists(face_dir):
        face_files = [f for f in os.listdir(face_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]  # i have used only these image formats,if you have other formats please add them here.
        
        for img_file in tqdm(face_files, desc="Enhancing Face images"):
            if face_count >= MAX_IMAGES_FACES:
                break
                
            img_path = os.path.join(face_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Apply enhancement
            enhanced_img = enhance_image(img)
            
            # Save enhanced image
            save_path = os.path.join(enhanced_face_dir, img_file)
            cv2.imwrite(save_path, enhanced_img)
            face_count += 1

    # Process non-face images
    if os.path.exists(nonface_dir):
        nonface_files = [f for f in os.listdir(nonface_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]  # i have used only these image formats,if you have other formats please add them here.
        
        for img_file in tqdm(nonface_files, desc="Enhancing Non-Face images"):
            if nonface_count >= MAX_IMAGES:
                break
                
            img_path = os.path.join(nonface_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Apply enhancement
            enhanced_img = enhance_image(img)
            
            # Save enhanced image
            save_path = os.path.join(enhanced_nonface_dir, img_file)
            cv2.imwrite(save_path, enhanced_img)
            nonface_count += 1

    print(f"Enhanced dataset created:")
    print(f"  - Enhanced face images: {face_count}")
    print(f"  - Enhanced non-face images: {nonface_count}")


# 4. ROI Dataset Builder (Using Enhanced Images)

def prepare_roi_dataset(enhanced_face_dir, enhanced_nonface_dir, roi_face_dir, roi_nonface_dir, roi_size=(128, 128)):
    """
    Extract ROIs from enhanced images and save them in separate directories as you need.
    - For faces: Use Haar cascade to detect face, then extract and resize ROI(128x128)
    - For non-faces: Use center region as ROI and resize to roi_size (Because, non-faces won't have detectable faces)
    """
    
    # Clear and recreate ROI directories whenever this function is called
    for directory in [roi_face_dir, roi_nonface_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
    
    face_count, nonface_count = 0, 0
    face_skipped, nonface_processed = 0, 0

    # Process enhanced face images
    if os.path.exists(enhanced_face_dir):
        face_files = [f for f in os.listdir(enhanced_face_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in tqdm(face_files, desc="Extracting Face ROIs from enhanced images"):
            if face_count >= MAX_IMAGES:
                break
                
            img_path = os.path.join(enhanced_face_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue
                
            # Use improved parameters for face detection on enhanced image
            faces = face_cascade.detectMultiScale(
                img,  # Already grayscale and enhanced image
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) > 0:
                # Take the largest detected face (i.e., most prominent as per area and nearest to camera)
                x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
                roi = img[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, roi_size, interpolation=cv2.INTER_AREA)
                
                # Save ROI with same filename (to keep track and ease mapping)
                save_path = os.path.join(roi_face_dir, img_file)
                cv2.imwrite(save_path, roi_resized)
                face_count += 1
            else:
                face_skipped += 1

    # Process enhanced non-face images
    if os.path.exists(enhanced_nonface_dir):
        nonface_files = [f for f in os.listdir(enhanced_nonface_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for img_file in tqdm(nonface_files, desc="Processing Enhanced Non-Face images"):
            if nonface_count >= MAX_IMAGES:
                break
                
            img_path = os.path.join(enhanced_nonface_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            # Use center region as ROI for non-face images
            h, w = img.shape
            center_size = min(h, w) // 2  # Use half of smaller dimension
            y_center = h // 2
            x_center = w // 2
            y_start = max(0, y_center - center_size // 2)
            x_start = max(0, x_center - center_size // 2)
            y_end = min(h, y_start + center_size)
            x_end = min(w, x_start + center_size)
            
            roi = img[y_start:y_end, x_start:x_end]
            roi_resized = cv2.resize(roi, roi_size, interpolation=cv2.INTER_AREA)
            
            # Save with same filename
            save_path = os.path.join(roi_nonface_dir, img_file)
            cv2.imwrite(save_path, roi_resized)
            nonface_count += 1

    print(f"ROI dataset created from enhanced images:")
    print(f"  - Face ROIs extracted: {face_count}")
    print(f"  - Face images skipped (no face detected): {face_skipped}")
    print(f"  - Non-face center ROIs processed: {nonface_count}")


# 5. ROI Dataset Loader

def load_roi_dataset(roi_face_dir, roi_nonface_dir):
    """
    Load the pre-processed ROI images for training
    """
    X, y = [], []

    # Load face ROIs
    if os.path.exists(roi_face_dir):
        face_files = [f for f in os.listdir(roi_face_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for img_file in tqdm(face_files, desc=f"Loading Face ROIs"):
            img_path = os.path.join(roi_face_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Normalize to [0, 1]
            X.append(img.flatten().astype(np.float32) / 255.0)
            y.append(1)  # Face label

    # Load non-face ROIs
    if os.path.exists(roi_nonface_dir):
        nonface_files = [f for f in os.listdir(roi_nonface_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for img_file in tqdm(nonface_files, desc=f"Loading Non-Face ROIs"):
            img_path = os.path.join(roi_nonface_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # Normalize to [0, 1]
            X.append(img.flatten().astype(np.float32) / 255.0)
            y.append(0)  # Non-face label

    face_count = sum(1 for label in y if label == 1)
    nonface_count = sum(1 for label in y if label == 0)
    
    print(f"Final ROI dataset loaded:")
    print(f"  - Face ROIs: {face_count}")
    print(f"  - Non-face ROIs: {nonface_count}")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Feature dimension: {len(X[0]) if X else 0}")

    return np.array(X), np.array(y)


# 6. Quantum Detection Model (VQC) - 8 QUBITS

dev = qml.device("default.qubit", wires=N_QUBITS) # Define the quantum device with 8 qubits ,you can change the device to gpu also if you have a compatible GPU.

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

class VQC(nn.Module):
    def __init__(self):
        super().__init__()
        weight_shapes = {"weights": (3, N_QUBITS, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.fc = nn.Linear(2, 2)  # Optional final linear layer for classification This will be used if you want to add a final linear layer after quantum layer for better classification.
    
    def forward(self, x):
        return self.fc(self.qlayer(x))   # Optional final linear layer for classification 

def train_vqc_with_validation(X_train, y_train, X_test, y_test, epochs=100):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    model = VQC()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        opt.step()

        train_pred = torch.argmax(outputs, axis=1)
        train_acc = (train_pred == y_train).float().mean().item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = loss_fn(val_outputs, y_test)
            val_pred = torch.argmax(val_outputs, axis=1)
            val_acc = (val_pred == y_test).float().mean().item()

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred_final = torch.argmax(model(X_test), axis=1).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        cm = confusion_matrix(y_test_np, val_pred_final, labels=[0,1], normalize='true')
        print("\nNormalized Confusion Matrix (Validation Set):\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", 
                xticklabels=["Non-Face", "Face"], yticklabels=["Non-Face", "Face"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Normalized Confusion Matrix (VQC Validation)")
    plt.show()

    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training vs Validation Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title("Training vs Validation Accuracy")
    
    plt.tight_layout()
    plt.show()

    # Save model
    torch.save(model.state_dict(), "vqc_face_model_roi.pth")  # Again you can change the model name as per your need.
    return model


# 7. Enhanced Main Training Pipeline - DETECTION ONLY

def main_pipeline():
    print("Starting ENHANCED ROI-based Quantum Face Detection System\n" + "="*60)
    print("USING 8 QUBITS FOR FACE DETECTION")
    print("="*60)

    # Step 1: Prepare Enhanced datasets (Faces, Non-faces)
    print("PART 1: Preparing Enhanced Datasets\n" + "-"*40)
    start_enhance_prep = time.time()
    
    # Enhance face and non-face datasets
    prepare_enhanced_dataset(FACE_DIR, NONFACE_DIR, ENHANCED_FACE_DIR, ENHANCED_NONFACE_DIR)
    
    enhance_prep_time = time.time() - start_enhance_prep
    print(f"Enhanced datasets preparation time: {enhance_prep_time:.4f} seconds")

    # Step 2: Prepare ROI datasets from enhanced images (Faces, Non-faces)
    print("PART 2: Preparing ROI Datasets from Enhanced Images\n" + "-"*40)
    start_roi_prep = time.time()
    
    # Prepare face and non-face ROIs for detection training
    prepare_roi_dataset(ENHANCED_FACE_DIR, ENHANCED_NONFACE_DIR, ROI_FACE_DIR, ROI_NONFACE_DIR, ROI_SIZE)
    
    roi_prep_time = time.time() - start_roi_prep
    print(f"ROI datasets preparation time: {roi_prep_time:.4f} seconds")

    # Step 3: Load ROI dataset for detection training
    print(f"\nPART 3: Loading ROI Dataset for Detection Training\n" + "-"*40)
    start_load = time.time()
    X, y = load_roi_dataset(ROI_FACE_DIR, ROI_NONFACE_DIR)
    if len(X) == 0:
        print("No ROI data loaded. Exiting...")
        return None
    load_time = time.time() - start_load
    print(f"ROI dataset load time: {load_time:.4f} seconds")

    # Step 4: PCA transformation for detection
    print(f"\nPART 4: PCA Transformation for Detection\n" + "-"*40)
    start_pca = time.time()
    pca_detection = PCA(n_components=N_QUBITS, random_state=42)
    X_reduced = pca_detection.fit_transform(X)
    X_reduced = (X_reduced - X_reduced.min()) / (X_reduced.max() - X_reduced.min())
    X_reduced = X_reduced * 2 * np.pi
    pca_time = time.time() - start_pca
    print(f"PCA processing time: {pca_time:.4f} seconds")
    print(f"Reduced feature dimension: {X_reduced.shape[1]}")

    # Save PCA model
    joblib.dump(pca_detection, "pca_detection_roi.pkl") # You can change the pca model name as per your need.

    # Step 5: Train-test split
    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train_q.shape[0]} samples")
    print(f"Test set: {X_test_q.shape[0]} samples")

    # Step 6: Train quantum model
    print(f"\nPART 5: Training Quantum Model (8 Qubits)\n" + "-"*40)
    start_training = time.time()
    vqc_model = train_vqc_with_validation(X_train_q, y_train_q, X_test_q, y_test_q, epochs=VQC_EPOCHS)
    training_time = time.time() - start_training
    print(f"Model training time: {training_time:.4f} seconds")

    total_pipeline_time = enhance_prep_time + roi_prep_time + load_time + pca_time + training_time
    print(f"\nTotal pipeline time: {total_pipeline_time:.4f} seconds")

    print("\nDETECTION-ONLY pipeline completed successfully!")
    print("Recognition database must be built separately using db_creation2.py")
    return {
        'vqc_model': vqc_model,
        'pca_detection': pca_detection
    }


if __name__ == "__main__":
    models = main_pipeline()