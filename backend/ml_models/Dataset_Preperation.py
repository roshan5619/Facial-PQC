import os
import shutil
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image


celeba_dir = "./celebA"
faces_dir = "./Dataset25k/faces"
nonface_dir = "./Dataset25k/nonfaces"


os.makedirs(celeba_dir, exist_ok=True)
os.makedirs(faces_dir, exist_ok=True)
os.makedirs(nonface_dir, exist_ok=True)

# Authenticate Kaggle API 
api = KaggleApi()
api.authenticate()

print("Downloading CelebA dataset...!")
api.dataset_download_files("jessicali9530/celeba-dataset", path=celeba_dir, unzip=True)

# # Path to celebA images
celeba_images = os.path.join(celeba_dir, "img_align_celeba/img_align_celeba")

# Copy 25k CelebA images into faces folder
print("Copying 25,000 CelebA face images...")
for img_name in tqdm(sorted(os.listdir(celeba_images))[:35000]):
    src = os.path.join(celeba_images, img_name)
    dst = os.path.join(faces_dir, img_name)
    shutil.copyfile(src, dst)



print("Downloading CIFAR-10 dataset...")

# Non-face classes to extract
nonface_classes = ['airplane', 'automobile', 'ship', 'truck', 'frog']

# Transformation: resize + grayscale
transform = transforms.Compose([
    transforms.Resize((48, 48)),   # match CelebA faces
    transforms.Grayscale()
])

# Load CIFAR-10 (train split)
cifar = CIFAR10(root="./", download=True, train=True)

# Get class names
class_names = cifar.classes

# Save up to 25,000 CIFAR-10 non-face images
print("Extracting 25,000 non-face images...")
saved = 0
for i in range(len(cifar)):
    label = class_names[cifar.targets[i]]
    if label in nonface_classes:
        img = transform(cifar[i][0])  # apply transform
        img.save(os.path.join(nonface_dir, f"nonface_{saved:05d}.png"))
        saved += 1
    if saved >= 35000:
        break

print("Dataset preparation complete!")
print(f"Faces saved in: {faces_dir}")
print(f"Non-faces saved in: {nonface_dir}")

