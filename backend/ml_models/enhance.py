import os
import cv2
import numpy as np
from tqdm import tqdm

# ========================
# Enhancement parameters
# ========================
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
GAMMA_CORRECTION = 1.2
ROI_SIZE = (128, 128)  # Resize ROI to this size

# ========================
# Hardcoded Paths
# ========================
INPUT_DIR = r"Dataset25k\qc_class"     # <-- Replace with your input folder
OUTPUT_DIR = r"Dataset25k\qc_class_roi"     # <-- Replace with folder to save enhanced face ROIs

# ========================
# Load Haar Cascade
# ========================
haar_paths = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_default.xml",
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
    raise RuntimeError("Failed to load Haar cascade. Please ensure the file exists.")

# ========================
# Enhancement Functions
# ========================
def gamma_correction(image, gamma=1.0):
    """Apply gamma correction to normalize brightness."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_image(image):
    """Enhance image with grayscale, CLAHE, and gamma correction."""
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(gray)
    
    # Gamma correction
    enhanced = gamma_correction(enhanced, gamma=GAMMA_CORRECTION)
    
    return enhanced

# ========================
# Main ROI Enhancement Pipeline
# ========================
def enhance_face_rois(input_dir: str, output_dir: str):
    """Detect face ROIs, enhance them, and save to output directory."""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    face_count = 0
    skipped_count = 0

    for img_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)

        # Skip existing files
        if os.path.exists(output_path):
            continue

        img = cv2.imread(input_path)
        if img is None:
            skipped_count += 1
            continue

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        
        if len(faces) == 0:
            skipped_count += 1
            continue

        # Take largest face
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        roi = img[y:y+h, x:x+w]

        # Resize ROI
        roi_resized = cv2.resize(roi, ROI_SIZE, interpolation=cv2.INTER_AREA)

        # Enhance
        enhanced_roi = enhance_image(roi_resized)

        # Save enhanced ROI
        cv2.imwrite(output_path, enhanced_roi)
        face_count += 1

    print(f"\nProcessing completed.")
    print(f"  - Enhanced face ROIs saved: {face_count}")
    print(f"  - Images skipped (no face or read error): {skipped_count}")
    print(f"Enhanced ROIs saved to: {output_dir}")

# ========================
# Script Entry
# ========================
if __name__ == "__main__":
    enhance_face_rois(INPUT_DIR, OUTPUT_DIR)
