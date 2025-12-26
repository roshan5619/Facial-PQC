"""
Script to migrate your existing users when
images have NO pattern (e.g. WIN_20250929_17_52_50_Pro.jpg).

This script groups images in batches of 3:
    3 images = 1 user
"""

import sys
import os
import glob

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.database import SessionLocal
from app.db.crud import UserCRUD
from app.services.cloudinary_service import cloudinary_service


def migrate_users():
    db = SessionLocal()

    images_dir = "data/existing_users"  # <-- your path
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    if len(image_paths) == 0:
        print("❌ No JPG images found in data/existing_users")
        return

    if len(image_paths) % 3 != 0:
        print(f"⚠️ WARNING: Found {len(image_paths)} images, not divisible by 3")
        print("    Extra images will be ignored.")

    print(f"📸 Total images found: {len(image_paths)}")
    total_users = len(image_paths) // 3
    print(f"👤 Total users to create: {total_users}")

    user_index = 1
    i = 0

    while i + 2 < len(image_paths):
        user_imgs = image_paths[i:i + 3]
        i += 3

        print(f"\n➡️ User {user_index}:")
        print(f"    Files: {os.path.basename(user_imgs[0])}, {os.path.basename(user_imgs[1])}, {os.path.basename(user_imgs[2])}")

        image_urls = []

        # Upload each of the 3 images
        for img_num, img_path in enumerate(user_imgs, start=1):
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()

                result = cloudinary_service.upload_enrollment_image(
                    img_bytes,
                    user_index,
                    img_num,
                    f"migrated_user_{user_index}"
                )

                if result["success"]:
                    image_urls.append(result["url"])
                else:
                    raise Exception("Cloudinary upload failed")

            except Exception as e:
                print(f"❌ Error uploading image {img_num} for user {user_index}: {e}")
                break

        if len(image_urls) != 3:
            print(f"⚠️ Skipping user {user_index} due to upload issues")
            continue

        # Insert user into DB
        try:
            user = UserCRUD.create_user(
                db,
                username=f"user_{user_index}",
                email=f"user_{user_index}@demo.com",
                full_name=f"Demo User {user_index}",
                image1_url=image_urls[0],
                image2_url=image_urls[1],
                image3_url=image_urls[2]
            )

            print(f"✅ User created: {user.username}")

        except Exception as e:
            print(f"❌ Database error for user {user_index}: {e}")
            continue

        user_index += 1

    db.close()
    print("\n🎉 Migration complete!")
    print("➡️ Next: run db_creation2.py to rebuild your recognition database")


if __name__ == "__main__":
    migrate_users()
