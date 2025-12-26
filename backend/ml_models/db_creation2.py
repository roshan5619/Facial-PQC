# import os
# import cv2
# import time
# import numpy as np
# import joblib
# from tqdm import tqdm
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import sys
# sys.stdout.reconfigure(encoding='utf-8')

# # ========================
# # CONFIGURATION
# # ========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# RECOGNITION_DATASET_DIR = os.path.join(BASE_DIR, "Dataset25k", "college_ROI")
# ROI_SIZE = (128, 128)
# N_PCA_COMPONENTS = 128

# # Enhancement parameters
# CLAHE_CLIP_LIMIT = 2.0
# CLAHE_GRID_SIZE = (8, 8)
# GAMMA_CORRECTION = 1.2

# # Output file prefixes
# OUTPUT_PREFIX = "new_"

# # ========================
# # 1. Load Haar Cascade
# # ========================
# haar_paths = [
#     os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
#     os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"),
#     "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
# ]

# face_cascade = None
# for path in haar_paths:
#     if os.path.exists(path):
#         face_cascade = cv2.CascadeClassifier(path)
#         if not face_cascade.empty():
#             print(f"Loaded Haar cascade from: {path}")
#             break

# if face_cascade is None or face_cascade.empty():
#     raise RuntimeError("Failed to load Haar cascade.")

# # ========================
# # 2. Image Enhancement Functions
# # ========================
# def enhance_image(image):
#     """Apply enhancement pipeline to normalize image style"""
#     # Convert to grayscale if needed
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image.copy()
    
#     # Apply CLAHE for contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
#     enhanced = clahe.apply(gray)
    
#     # Apply gamma correction for brightness normalization
#     enhanced = gamma_correction(enhanced, gamma=GAMMA_CORRECTION)
    
#     return enhanced

# def gamma_correction(image, gamma=1.0):
#     """Apply gamma correction to normalize brightness"""
#     inv_gamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(image, table)

# # ========================
# # 3. Enhanced Face Detection for Recognition
# # ========================
# def safe_imread(path: str):
#     """Safe image reading with multiple fallbacks"""
#     try:
#         data = np.fromfile(path, dtype=np.uint8)
#         img = cv2.imdecode(data, cv2.IMREAD_COLOR)
#         if img is None:
#             img = cv2.imread(path)
#     except Exception:
#         img = cv2.imread(path)
#     return img

# def detect_face_roi_recognition_enhanced(img_bgr: np.ndarray, fallback_center: bool = True):
#     """Enhanced version with image preprocessing before face detection"""
#     # Apply enhancement before face detection
#     enhanced_img = enhance_image(img_bgr)
    
#     h0, w0 = enhanced_img.shape[:2]
    
#     # Use improved parameters for face detection
#     faces = face_cascade.detectMultiScale(
#         enhanced_img, 
#         scaleFactor=1.1, 
#         minNeighbors=5, 
#         minSize=(30,30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
    
#     if len(faces) > 0:
#         x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
#     else:
#         if not fallback_center:
#             return None, (0,0,0,0)
#         side = min(w0, h0)
#         x = (w0 - side) // 2
#         y = (h0 - side) // 2
#         w = h = side
    
#     roi = enhanced_img[y:y+h, x:x+w]
#     roi_resized = cv2.resize(roi, ROI_SIZE, interpolation=cv2.INTER_AREA)
#     return roi_resized, (x,y,w,h)

# def image_to_vector(img_gray: np.ndarray) -> np.ndarray:
#     """Convert image to normalized feature vector"""
#     v = img_gray.astype(np.float32).ravel() / 255.0
#     return v

# # ========================
# # 4. Enhanced Recognition Database Builder
# # ========================
# def build_face_dataset_recognition_enhanced(dir_path: str):
#     """Build enhanced recognition dataset with preprocessing"""
#     paths = []
#     if not os.path.exists(dir_path):
#         print(f"ERROR: Recognition dataset directory {dir_path} does not exist")
#         print("Please create the Dataset25k/college_ROI directory and add face images")
#         return np.zeros((0, ROI_SIZE[0]*ROI_SIZE[1])), []
        
#     # Collect all image paths
#     for ext in (".jpg", ".jpeg", ".png", ".bmp"):
#         for root, _, files in os.walk(dir_path):
#             for f in files:
#                 if f.lower().endswith(ext):
#                     paths.append(os.path.join(root, f))
    
#     paths = sorted(paths)
#     X = []
#     used_paths = []
#     skipped_count = 0
    
#     print(f"Processing {len(paths)} images from {dir_path}")
    
#     for p in tqdm(paths, desc="Processing recognition images"):
#         img = safe_imread(p)
#         if img is None:
#             skipped_count += 1
#             continue
        
#         roi, bbox = detect_face_roi_recognition_enhanced(img, fallback_center=True)
#         if roi is None:
#             skipped_count += 1
#             continue
        
#         X.append(image_to_vector(roi))
#         used_paths.append(p)
    
#     if len(X) == 0:
#         print("ERROR: No faces were processed from the recognition dataset")
#         return np.zeros((0, ROI_SIZE[0]*ROI_SIZE[1])), []
    
#     print(f"Successfully processed {len(X)} face images")
#     if skipped_count > 0:
#         print(f"Skipped {skipped_count} images (unreadable or no face detected)")
    
#     return np.vstack(X), used_paths

# def build_recognition_database_pca(X: np.ndarray, n_components: int = 16):
#     """Build PCA-based recognition database"""
#     print(f"Building PCA database with {n_components} components...")
    
#     # Standardization
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # PCA transformation
#     pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]), random_state=42)
#     feats_pca = pca.fit_transform(X_scaled)
    
#     print(f"PCA Database Summary:")
#     print(f"  Original dimensions: {X.shape[1]}")
#     print(f"  PCA dimensions: {feats_pca.shape[1]}")
#     print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
#     return feats_pca, scaler, pca

# # ========================
# # 5. Main Database Building Function
# # ========================
# def build_recognition_database():
#     """Main function to build the recognition database"""
#     print("=" * 70)
#     print("RECOGNITION DATABASE BUILDER")
#     print("=" * 70)
    
#     # Check if input directory exists
#     if not os.path.exists(RECOGNITION_DATASET_DIR):
#         print(f"Creating directory: {RECOGNITION_DATASET_DIR}")
#         os.makedirs(RECOGNITION_DATASET_DIR, exist_ok=True)
#         print(f"Please add face images to {RECOGNITION_DATASET_DIR} and run this script again.")
#         return False
    
#     # Step 1: Build enhanced face dataset
#     print("\nSTEP 1: Building Enhanced Face Dataset")
#     print("-" * 40)
#     start_time = time.time()
    
#     X_recognition, used_paths = build_face_dataset_recognition_enhanced(RECOGNITION_DATASET_DIR)
    
#     if X_recognition.shape[0] == 0:
#         print("ERROR: No face data available for database creation")
#         return False
    
#     dataset_time = time.time() - start_time
#     print(f"Dataset building time: {dataset_time:.4f} seconds")
#     print(f"Recognition dataset: {len(used_paths)} images, vector dim: {X_recognition.shape[1]}")
    
#     # Step 2: Build PCA database
#     print("\nSTEP 2: Building PCA Database")
#     print("-" * 40)
#     start_time = time.time()
    
#     recognition_db, scaler_recognition, pca_recognition = build_recognition_database_pca(
#         X_recognition, n_components=N_PCA_COMPONENTS)
    
#     pca_time = time.time() - start_time
#     print(f"PCA processing time: {pca_time:.4f} seconds")
    
#     # Step 3: Save database files
#     print("\nSTEP 3: Saving Database Files")
#     print("-" * 40)
    
#     # Save with "new_" prefix
#     pca_file = f"{OUTPUT_PREFIX}pca_recognition_cosine_roi.pkl"
#     scaler_file = f"{OUTPUT_PREFIX}scaler_recognition_cosine_roi.pkl"
#     db_file = f"{OUTPUT_PREFIX}recognition_db_cosine_roi.pkl"
#     paths_file = f"{OUTPUT_PREFIX}used_paths_cosine_roi.pkl"
    
#     joblib.dump(pca_recognition, pca_file)
#     joblib.dump(scaler_recognition, scaler_file)
#     joblib.dump(recognition_db, db_file)
#     joblib.dump(used_paths, paths_file)
    
#     print(f"Saved database files with prefix '{OUTPUT_PREFIX}':")
#     print(f"  - {pca_file}")
#     print(f"  - {scaler_file}")
#     print(f"  - {db_file}")
#     print(f"  - {paths_file}")
    
#     total_time = dataset_time + pca_time
#     print(f"\nTotal database building time: {total_time:.4f} seconds")
    
#     # Summary
#     print("\n" + "=" * 70)
#     print("DATABASE BUILDING COMPLETED SUCCESSFULLY!")
#     print("=" * 70)
#     print(f"Database contains {len(used_paths)} face embeddings")
#     print(f"PCA dimensions: {recognition_db.shape[1]}")
#     print(f"Files are ready for use in the main training pipeline")
    
#     return True

# # ========================
# # 6. Main Execution
# # ========================
# if __name__ == "__main__":
#     print("Starting Recognition Database Builder...")
#     print("This script creates PCA-based face recognition database from college_ROI images")
#     print()
    
#     success = build_recognition_database()
    
#     if success:
#         print("\nYou can now run the main training pipeline with the new database.")
#     else:
#         print("\nDatabase building failed. Please check the error messages above.")

#=========================================================
import os
import cv2
import time
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ========================
# CONFIGURATION
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECOGNITION_DATASET_DIR = os.path.join(BASE_DIR, "Dataset25k", "qc_class_roi")
ROI_OUTPUT_DIR = os.path.join(BASE_DIR, "Dataset25k", "qc_class_roi")  # ROI storage and recognition directory
ROI_SIZE = (128, 128)
N_PCA_COMPONENTS = 512

# Enhancement parameters (SAME AS PREDICTION)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAMMA_CORRECTION = 1.2

# Output file prefixes
OUTPUT_PREFIX = "qcclass_"

# ========================
# 1. Load Haar Cascade
# ========================
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

# ========================
# 2. Image Enhancement Functions (SAME AS PREDICTION)
# ========================
def enhance_image(image):
    """Apply enhancement pipeline to normalize image style - SAME AS PREDICTION"""
    # Convert to grayscale if needed
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
    """Apply gamma correction to normalize brightness - SAME AS PREDICTION"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_roi(roi_image):
    """Apply enhancement to ROI image specifically - SAME AS PREDICTION"""
    # ROI is already grayscale, just apply CLAHE and gamma
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(roi_image)
    enhanced = gamma_correction(enhanced, gamma=GAMMA_CORRECTION)
    return enhanced

# ========================
# 3. Enhanced Face Detection for Recognition
# ========================
def safe_imread(path: str):
    """Safe image reading with multiple fallbacks"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(path)
    except Exception:
        img = cv2.imread(path)
    return img

def extract_roi_for_database(img_bgr: np.ndarray):
    """Extract ROI for database with same enhancement as prediction"""
    # Apply enhancement to entire image first - SAME AS PREDICTION
    enhanced_img = enhance_image(img_bgr)
    
    # Detect faces on enhanced image - SAME AS PREDICTION
    faces = face_cascade.detectMultiScale(
        enhanced_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # ONLY USE IMAGES WITH FACE DETECTED - SKIP IF NO FACE
    if len(faces) == 0:
        return None, None
    
    # Take the largest detected face
    x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    roi = enhanced_img[y:y+h, x:x+w]
    
    # Resize ROI
    roi_resized = cv2.resize(roi, ROI_SIZE, interpolation=cv2.INTER_AREA)
    
    # Apply separate enhancement to ROI - SAME AS PREDICTION
    roi_enhanced = enhance_roi(roi_resized)
    
    return roi_enhanced, (x, y, w, h)

def image_to_vector(img_gray: np.ndarray) -> np.ndarray:
    """Convert image to normalized feature vector"""
    v = img_gray.astype(np.float32).ravel() / 255.0
    return v

# ========================
# 4. Enhanced Recognition Database Builder
# ========================
def build_face_dataset_from_roi_dir(roi_dir: str):
    """Build recognition dataset from ROI directory - UPDATED: Use ROI directory directly"""
    paths = []
    if not os.path.exists(roi_dir):
        print(f"ERROR: ROI directory {roi_dir} does not exist")
        print("Please run the ROI extraction process first")
        return np.zeros((0, ROI_SIZE[0]*ROI_SIZE[1])), []
        
    # Collect all ROI image paths
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        for root, _, files in os.walk(roi_dir):
            for f in files:
                if f.lower().endswith(ext):
                    paths.append(os.path.join(root, f))
    
    paths = sorted(paths)
    X = []
    used_paths = []
    skipped_count = 0
    
    print(f"Processing {len(paths)} ROI images from {roi_dir}")
    
    for p in tqdm(paths, desc="Processing ROI images for recognition"):
        # Read ROI image directly (already enhanced and processed)
        roi_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if roi_img is None:
            skipped_count += 1
            continue
        
        # ROI images are already the correct size and enhanced
        # Just convert to vector for PCA
        X.append(image_to_vector(roi_img))
        used_paths.append(p)
    
    if len(X) == 0:
        print("ERROR: No ROI images were processed for recognition database")
        return np.zeros((0, ROI_SIZE[0]*ROI_SIZE[1])), []
    
    print(f"Successfully processed {len(X)} ROI images for recognition database")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} ROI images (unreadable)")
    
    return np.vstack(X), used_paths

def build_face_dataset_recognition_enhanced(dir_path: str, roi_output_dir: str):
    """Build enhanced recognition dataset with preprocessing and save ROI images"""
    paths = []
    if not os.path.exists(dir_path):
        print(f"ERROR: Recognition dataset directory {dir_path} does not exist")
        print("Please create the Dataset25k/college directory and add face images")
        return np.zeros((0, ROI_SIZE[0]*ROI_SIZE[1])), [], []
    
    # Create ROI output directory
    os.makedirs(roi_output_dir, exist_ok=True)
    print(f"Created ROI output directory: {roi_output_dir}")
        
    # Collect all image paths
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.lower().endswith(ext):
                    paths.append(os.path.join(root, f))
    
    paths = sorted(paths)
    X = []
    used_paths = []
    saved_roi_paths = []  # Store paths to saved ROI images
    skipped_count = 0
    face_not_found_count = 0
    roi_saved_count = 0
    
    print(f"Processing {len(paths)} images from {dir_path}")
    
    for i, p in enumerate(tqdm(paths, desc="Processing recognition images")):
        img = safe_imread(p)
        if img is None:
            skipped_count += 1
            continue
        
        # Extract ROI with same enhancement as prediction
        roi, bbox = extract_roi_for_database(img)
        
        # SKIP IMAGE IF NO FACE DETECTED
        if roi is None:
            face_not_found_count += 1
            continue
        
        # Save ROI image with serial naming
        roi_filename = f"roi_{i+1:05d}.png"
        roi_save_path = os.path.join(roi_output_dir, roi_filename)
        cv2.imwrite(roi_save_path, roi)
        saved_roi_paths.append(roi_save_path)
        roi_saved_count += 1
        
        # Create feature vector for PCA
        X.append(image_to_vector(roi))
        used_paths.append(p)
    
    if len(X) == 0:
        print("ERROR: No faces were processed from the recognition dataset")
        return np.zeros((0, ROI_SIZE[0]*ROI_SIZE[1])), [], []
    
    print(f"Successfully processed {len(X)} face images")
    print(f"ROI images saved: {roi_saved_count}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} images (unreadable)")
    if face_not_found_count > 0:
        print(f"Skipped {face_not_found_count} images (no face detected)")
    
    return np.vstack(X), used_paths, saved_roi_paths

def build_recognition_database_pca(X: np.ndarray, n_components: int = 512):
    """Build PCA-based recognition database"""
    print(f"Building PCA database with {n_components} components...")
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA transformation
    pca = PCA(n_components=min(n_components, X_scaled.shape[0], X_scaled.shape[1]), random_state=42)
    feats_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA Database Summary:")
    print(f"  Original dimensions: {X.shape[1]}")
    print(f"  PCA dimensions: {feats_pca.shape[1]}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    return feats_pca, scaler, pca

# ========================
# 5. Main Database Building Function
# ========================
def build_recognition_database():
    """Main function to build the recognition database"""
    print("=" * 70)
    print("RECOGNITION DATABASE BUILDER (512 PCA COMPONENTS)")
    print("=" * 70)
    print("Features:")
    print("- Same enhancement pipeline as prediction")
    print("- Only images with detected faces are used")
    print("- ROI images saved to College_roi folder with serial naming")
    print("- PCA embeddings created from enhanced ROIs")
    print("- Database uses ROI directory for recognition")
    print("=" * 70)
    
    # Check if ROI directory already exists with images
    roi_images_exist = False
    if os.path.exists(ROI_OUTPUT_DIR):
        roi_files = [f for f in os.listdir(ROI_OUTPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        roi_images_exist = len(roi_files) > 0
    
    if roi_images_exist:
        print(f"ROI directory {ROI_OUTPUT_DIR} already contains {len(roi_files)} images")
        print("Using existing ROI images for database creation...")
        
        # Step 1: Build dataset from existing ROI directory
        print("\nSTEP 1: Building Dataset from Existing ROI Images")
        print("-" * 50)
        start_time = time.time()
        
        X_recognition, used_paths = build_face_dataset_from_roi_dir(ROI_OUTPUT_DIR)
        
        if X_recognition.shape[0] == 0:
            print("ERROR: No ROI data available for database creation")
            return False
        
        dataset_time = time.time() - start_time
        print(f"Dataset building time: {dataset_time:.4f} seconds")
        print(f"Recognition dataset: {len(used_paths)} ROI images, vector dim: {X_recognition.shape[1]}")
        
        saved_roi_paths = used_paths  # For consistency with the rest of the code
        
    else:
        # Check if input directory exists
        if not os.path.exists(RECOGNITION_DATASET_DIR):
            print(f"Creating directory: {RECOGNITION_DATASET_DIR}")
            os.makedirs(RECOGNITION_DATASET_DIR, exist_ok=True)
            print(f"Please add face images to {RECOGNITION_DATASET_DIR} and run this script again.")
            return False
        
        # Step 1: Build enhanced face dataset and save ROI images
        print("\nSTEP 1: Building Enhanced Face Dataset & Saving ROI Images")
        print("-" * 50)
        start_time = time.time()
        
        X_recognition, used_paths, saved_roi_paths = build_face_dataset_recognition_enhanced(
            RECOGNITION_DATASET_DIR, ROI_OUTPUT_DIR)
        
        if X_recognition.shape[0] == 0:
            print("ERROR: No face data available for database creation")
            return False
        
        dataset_time = time.time() - start_time
        print(f"Dataset building time: {dataset_time:.4f} seconds")
        print(f"Recognition dataset: {len(used_paths)} images, vector dim: {X_recognition.shape[1]}")
        print(f"ROI images saved to: {ROI_OUTPUT_DIR}")
    
    # Step 2: Build PCA database
    print("\nSTEP 2: Building PCA Database")
    print("-" * 40)
    start_time = time.time()
    
    recognition_db, scaler_recognition, pca_recognition = build_recognition_database_pca(
        X_recognition, n_components=N_PCA_COMPONENTS)
    
    pca_time = time.time() - start_time
    print(f"PCA processing time: {pca_time:.4f} seconds")
    
    # Step 3: Save database files
    print("\nSTEP 3: Saving Database Files")
    print("-" * 40)
    
    # Save with "new_" prefix
    pca_file = f"{OUTPUT_PREFIX}pca_recognition_cosine_roi.pkl"
    scaler_file = f"{OUTPUT_PREFIX}scaler_recognition_cosine_roi.pkl"
    db_file = f"{OUTPUT_PREFIX}recognition_db_cosine_roi.pkl"
    paths_file = f"{OUTPUT_PREFIX}used_paths_cosine_roi.pkl"
    roi_paths_file = f"{OUTPUT_PREFIX}roi_paths_cosine_roi.pkl"
    
    joblib.dump(pca_recognition, pca_file)
    joblib.dump(scaler_recognition, scaler_file)
    joblib.dump(recognition_db, db_file)
    joblib.dump(used_paths, paths_file)
    joblib.dump(saved_roi_paths, roi_paths_file)
    
    print(f"Saved database files with prefix '{OUTPUT_PREFIX}':")
    print(f"  - {pca_file} (PCA model)")
    print(f"  - {scaler_file} (Scaler)")
    print(f"  - {db_file} (Database embeddings)")
    print(f"  - {paths_file} (Original image paths)")
    print(f"  - {roi_paths_file} (ROI image paths)")
    
    total_time = dataset_time + pca_time
    print(f"\nTotal database building time: {total_time:.4f} seconds")
    
    # Summary
    print("\n" + "=" * 70)
    print("DATABASE BUILDING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Database contains {len(used_paths)} face embeddings")
    print(f"PCA dimensions: {recognition_db.shape[1]}")
    print(f"ROI images available in: {ROI_OUTPUT_DIR}")
    print(f"Recognition database uses ROI directory for matching")
    print(f"Files are ready for use in prediction pipeline")
    
    return True

# ========================
# 6. Main Execution
# ========================
if __name__ == "__main__":
    print("Starting Recognition Database Builder...")
    print("This script creates PCA-based face recognition database")
    print(f"Using {N_PCA_COMPONENTS} PCA components for recognition")
    print(f"ROI directory: {ROI_OUTPUT_DIR}")
    print()
    
    success = build_recognition_database()
    
    if success:
        print("\nYou can now run the prediction pipeline with the new database.")
        print(f"ROI images are available in: {ROI_OUTPUT_DIR}")
        print("Database uses ROI directory for recognition matching")
    else:
        print("\nDatabase building failed. Please check the error messages above.")