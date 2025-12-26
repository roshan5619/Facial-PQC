
import os
import cv2
import time
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Recognition database files
RECOGNITION_DB_PREFIX = "qcclass_"   # Prefix for database files ,you can togle between qcclass_ and new_ as per your need.(or your own naming convention)

# Model parameters
N_QUBITS = 8
ROI_SIZE = (128, 128)
COSINE_THRESHOLD = 0.5

# Enhancement parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAMMA_CORRECTION = 1.2


# 1. Load Haar Cascade

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
    raise RuntimeError("Failed to load Haar cascade.")

# 2. Image Enhancement Functions

def enhance_image(image):
    """Apply enhancement pipeline to normalize image style"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(gray)
    enhanced = gamma_correction(enhanced, gamma=GAMMA_CORRECTION)
    return enhanced

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_roi(roi_image):
    """Apply enhancement to ROI image specifically"""
    # ROI is already grayscale, just apply CLAHE and gamma
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(roi_image)
    enhanced = gamma_correction(enhanced, gamma=GAMMA_CORRECTION)
    return enhanced


# 3. Quantum Model Definition

import pennylane as qml
import torch.nn as nn

dev = qml.device("default.qubit", wires=N_QUBITS)

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
        self.fc = nn.Linear(2, 2)
    
    def forward(self, x):
        return self.fc(self.qlayer(x))

# ========================
# 4. Load Pre-trained Models
# ========================
def load_models():
    print("Loading pre-trained models...")
    
    # Load detection models
    try:
        pca_detection = joblib.load("pca_detection_roi.pkl")
        vqc_model = VQC()
        vqc_model.load_state_dict(
            torch.load("vqc_face_model_roi.pth", 
                       map_location=torch.device("cpu"), 
                       weights_only=True)
        )
        vqc_model.eval()
        print("✓ Detection models loaded successfully")
    except Exception as e:
        print(f"✗ Error loading detection models: {e}")
        return None
    
    # Load recognition models
    try:
        pca_file = f"{RECOGNITION_DB_PREFIX}pca_recognition_cosine_roi.pkl"
        scaler_file = f"{RECOGNITION_DB_PREFIX}scaler_recognition_cosine_roi.pkl"
        db_file = f"{RECOGNITION_DB_PREFIX}recognition_db_cosine_roi.pkl"
        paths_file = f"{RECOGNITION_DB_PREFIX}used_paths_cosine_roi.pkl"
        
        pca_recognition = joblib.load(pca_file)
        scaler_recognition = joblib.load(scaler_file)
        recognition_db = joblib.load(db_file)
        used_paths = joblib.load(paths_file)
        
        print(f"✓ Recognition database loaded: {recognition_db.shape[0]} embeddings")
        print(f"✓ Recognition PCA dimensions: {recognition_db.shape[1]}")
    except Exception as e:
        print(f"✗ Error loading recognition database: {e}")
        print("Please run db_creation2.py to build the recognition database first.")
        return None
    
    return {
        'vqc_model': vqc_model,
        'pca_detection': pca_detection,
        'recognition_db': recognition_db,
        'scaler_recognition': scaler_recognition,
        'pca_recognition': pca_recognition,
        'used_paths': used_paths
    }


# 5. Enhanced Prediction Functions

def extract_detection_roi(img, roi_size=(128, 128)):
    """Extract ROI for detection with separate enhancement"""
    # Apply enhancement to entire image first
    enhanced_img = enhance_image(img)
    
    # Detect faces on enhanced image
    faces = face_cascade.detectMultiScale(
        enhanced_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        roi = enhanced_img[y:y+h, x:x+w]
    else:
        # Fallback to center crop  as no face detected
        h, w = enhanced_img.shape
        center_size = min(h, w) // 2
        y_center, x_center = h // 2, w // 2
        y_start, x_start = max(0, y_center - center_size // 2), max(0, x_center - center_size // 2)
        y_end, x_end = min(h, y_start + center_size), min(w, x_start + center_size)
        roi = enhanced_img[y_start:y_end, x_start:x_end]
        x, y, w, h = x_start, y_start, center_size, center_size
    
    # Resize ROI
    roi_resized = cv2.resize(roi, roi_size, interpolation=cv2.INTER_AREA)
    
    # Apply separate enhancement to ROI
    roi_enhanced = enhance_roi(roi_resized)
    
    # Normalize for detection
    roi_normalized = roi_enhanced.flatten().astype(np.float32) / 255.0
    
    return roi_normalized, (x, y, w, h), roi_enhanced

def extract_recognition_roi(img, roi_size=(128, 128)):
    """Extract ROI for recognition with separate enhancement"""
    # Apply enhancement to entire image first
    enhanced_img = enhance_image(img)
    
    # Detect faces on enhanced image
    faces = face_cascade.detectMultiScale(
        enhanced_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        roi = enhanced_img[y:y+h, x:x+w]
    else:
        # Fallback to center crop as no face detected
        h, w = enhanced_img.shape
        center_size = min(h, w) // 2
        y_center, x_center = h // 2, w // 2
        y_start, x_start = max(0, y_center - center_size // 2), max(0, x_center - center_size // 2)
        y_end, x_end = min(h, y_start + center_size), min(w, x_start + center_size)
        roi = enhanced_img[y_start:y_end, x_start:x_end]
        x, y, w, h = x_start, y_start, center_size, center_size
    
    # Resize ROI
    roi_resized = cv2.resize(roi, roi_size, interpolation=cv2.INTER_AREA)
    
    # Apply separate enhancement to ROI
    roi_enhanced = enhance_roi(roi_resized)
    
    return roi_enhanced, (x, y, w, h)

def match_face_cosine(probe_vec: np.ndarray, db_feats: np.ndarray, scaler, pca, threshold=0.7):
    probe_scaled = scaler.transform(probe_vec.reshape(1, -1))
    probe_pca = pca.transform(probe_scaled)
    similarities = cosine_similarity(probe_pca, db_feats)[0]
    best_idx = int(np.argmax(similarities))
    best_sim = similarities[best_idx]
    if best_sim < threshold:
        return -1, best_sim
    return best_idx, best_sim

def image_to_vector(img_gray: np.ndarray) -> np.ndarray:
    return img_gray.astype(np.float32).ravel() / 255.0


# 6. Display Functions

def display_comparison(probe_img, detection_roi, recognition_roi, matched_img_path, similarity, status, timing_info):
    plt.figure(figsize=(15, 10))
    
    # Original captured image
    plt.subplot(2, 4, 1)
    if len(probe_img.shape) == 3:
        plt.imshow(cv2.cvtColor(probe_img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(probe_img, cmap='gray')
    plt.title("Captured Image")
    plt.axis('off')
    
    # Detection ROI
    plt.subplot(2, 4, 2)
    plt.imshow(detection_roi, cmap='gray')
    plt.title("Detection ROI\n(8 PCA Components)")
    plt.axis('off')
    
    # Recognition ROI
    plt.subplot(2, 4, 3)
    plt.imshow(recognition_roi, cmap='gray')
    plt.title("Recognition ROI\n(512 PCA Components)")
    plt.axis('off')
    
    # Database match
    plt.subplot(2, 4, 4)
    if matched_img_path and os.path.exists(matched_img_path):
        matched_img = cv2.imread(matched_img_path)
        if matched_img is not None:
            plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Database Match\nSimilarity: {similarity:.4f}")
        else:
            plt.text(0.5, 0.5, "Image not found", ha='center', va='center', fontsize=12)
            plt.title("Database Match")
    else:
        plt.text(0.5, 0.5, "No match found", ha='center', va='center', fontsize=12, color='red')
        plt.title("Database Match")
    plt.axis('off')
    
    # Status information
    plt.subplot(2, 4, 5)
    plt.axis('off')
    status_text = f"Status: {status}\n\nSimilarity: {similarity:.4f}\nThreshold: {COSINE_THRESHOLD}\n\n"
    if "MATCH" in status:
        plt.text(0.1, 0.7, status_text, fontsize=12, va='center', color='green')
        if matched_img_path:
            plt.text(0.1, 0.4, f"Matched with:\n{os.path.basename(matched_img_path)}", fontsize=10, va='center')
    else:
        plt.text(0.1, 0.7, status_text, fontsize=12, va='center', color='red')
    
    # Timing information
    plt.subplot(2, 4, 6)
    plt.axis('off')
    timing_text = "Timing Information:\n\n" + "\n".join([f"{k}: {v:.4f}s" for k,v in timing_info.items()])
    plt.text(0.1, 0.7, timing_text, fontsize=11, va='center')
    
    # Similarity score
    plt.subplot(2, 4, 7)
    colors = ['red' if similarity < COSINE_THRESHOLD else 'green']
    plt.bar(['Similarity'], [similarity], color=colors)
    plt.axhline(y=COSINE_THRESHOLD, color='r', linestyle='--', label=f'Threshold: {COSINE_THRESHOLD}')
    plt.ylim(0, 1)
    plt.ylabel('Cosine Similarity')
    plt.title('Matching Score')
    plt.legend()
    
    # Processing pipeline
    plt.subplot(2, 4, 8)
    plt.axis('off')
    pipeline_text = "Processing Pipeline:\n\n1. Extract Detection ROI\n2. 8-PCA Detection\n3. Extract Recognition ROI\n4. 512-PCA Recognition\n5. Database Matching\n6. Result Display"
    plt.text(0.1, 0.5, pipeline_text, fontsize=10, va='center')
    
    plt.tight_layout()
    plt.show()

# 7. Webcam Prediction Function

def webcam_prediction(models):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nWebcam started. Press:")
    print("  SPACE - Capture and process image")
    print("  ESC   - Exit")
    print("  'a'   - Auto capture mode (every 3 seconds)")
    
    auto_capture = False
    last_capture_time = 0
    auto_interval = 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(display_frame, "SPACE: Capture | ESC: Exit | 'a': Auto", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if auto_capture:
            cv2.putText(display_frame, f"AUTO MODE: Next in {auto_interval - (time.time() - last_capture_time):.1f}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Face Recognition - Webcam", display_frame)
        
        current_time = time.time()
        if auto_capture and (current_time - last_capture_time) >= auto_interval:
            print(f"\nAuto-capturing image...")
            process_captured_image(frame, models)
            last_capture_time = current_time
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            print(f"\nManual capture triggered...")
            process_captured_image(frame, models)
        elif key == ord('a'):
            auto_capture = not auto_capture
            last_capture_time = current_time
            print(f"Auto capture mode {'ENABLED' if auto_capture else 'DISABLED'}")
    
    cap.release()
    cv2.destroyAllWindows()

# 8. Processing Captured Image

def process_captured_image(captured_frame, models):
    print("="*60)
    print("PROCESSING CAPTURED IMAGE")
    print("="*60)
    timing_info = {}
    
    # Extract ROI for Detection (with separate enhancement)
    start_time = time.time()
    detection_roi_vec, detection_bbox, detection_roi_img = extract_detection_roi(captured_frame, ROI_SIZE)
    timing_info['Detection ROI Extraction'] = time.time() - start_time
    
    if detection_roi_vec is None:
        print("Failed to extract ROI for detection")
        return
    
    # Face Detection with 8 PCA components
    start_time = time.time()
    features_pca = models['pca_detection'].transform([detection_roi_vec])
    features_norm = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min() + 1e-10)
    features_scaled = features_norm * 2 * np.pi
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    
    models['vqc_model'].eval()
    with torch.no_grad():
        outputs = models['vqc_model'](features_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = int(torch.argmax(outputs, dim=1).item())
        confidence = float(probs[0][pred].item())
    timing_info['Face Detection (8-PCA)'] = time.time() - start_time
    
    if pred == 0:
        print("RESULT: NON-FACE DETECTED")
        status, similarity, matched_img_path = "NON-FACE DETECTED", 0.0, None
        recognition_roi_img = None
    else:
        print("RESULT: FACE DETECTED - Proceeding to recognition...")
        
        # Extract ROI for Recognition (with separate enhancement)
        start_time = time.time()
        recognition_roi_img, recognition_bbox = extract_recognition_roi(captured_frame, ROI_SIZE)
        timing_info['Recognition ROI Extraction'] = time.time() - start_time
        
        if recognition_roi_img is None:
            print("Could not extract ROI for recognition")
            status, similarity, matched_img_path = "FACE DETECTED (ROI Fail)", 0.0, None
        else:
            # Convert ROI to vector for recognition (512 PCA components)
            start_time = time.time()
            probe_vec = image_to_vector(recognition_roi_img)
            timing_info['Feature Vector Creation'] = time.time() - start_time
            
            # Database Matching with 512 PCA components
            start_time = time.time()
            match_idx, best_sim = match_face_cosine(
                probe_vec, models['recognition_db'], 
                models['scaler_recognition'], models['pca_recognition'], 
                COSINE_THRESHOLD
            )
            timing_info['Database Matching (512-PCA)'] = time.time() - start_time
            
            if match_idx == -1:
                status, similarity, matched_img_path = "FACE DETECTED (No Match)", best_sim, None
            else:
                status, similarity, matched_img_path = "MATCH FOUND", best_sim, models['used_paths'][match_idx]
                print(f"MATCH: {os.path.basename(matched_img_path)} (similarity: {similarity:.4f})")
    
    timing_info['Total Processing'] = sum(timing_info.values())
    print("\nTiming Information:")
    for key, value in timing_info.items():
        print(f"  {key}: {value:.4f} seconds")
    
    # Display results
    display_comparison(captured_frame, detection_roi_img, recognition_roi_img, 
                      matched_img_path, similarity, status, timing_info)
    print("="*60)



if __name__ == "__main__":
    print("Webcam Face Recognition System")
    print("="*50)
    print("Using separate ROI extraction with:")
    print("- 8 PCA components for detection")
    print("- 512 PCA components for recognition")
    print("="*50)
    
    models = load_models()
    if models is None:
        print("Failed to load models. Please ensure:")
        print("1. You have run Face_DB2.py for detection training")
        print("2. You have run db_creation2.py for recognition database")
        print("3. Model files exist in the current directory")
        sys.exit(1)
    webcam_prediction(models)
    print("Webcam session ended.")