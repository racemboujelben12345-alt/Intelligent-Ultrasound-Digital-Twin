from pathlib import Path
from PIL import Image
from collections import Counter

ROOT = Path("data/raw/public_ultrasound/busi/Dataset_BUSI_with_GT")

classes = ["benign", "malignant", "normal"]

total_images = 0
total_masks = 0
errors = 0

print("=" * 60)
print("BUSI DATASET VALIDATION")
print("=" * 60)

for cls in classes:
    folder = ROOT / cls

    files = list(folder.glob("*.png"))

    images = [
        f for f in files
        if "_mask" not in f.stem
    ]

    masks = [
        f for f in files
        if "_mask" in f.stem
    ]

    print(f"\n{cls.upper()}")
    print("-" * 40)
    print(f"Images : {len(images)}")
    print(f"Masks  : {len(masks)}")

    total_images += len(images)
    total_masks += len(masks)

    sizes = Counter()
    mask_values = Counter()

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                sizes[img.size] += 1

        except Exception as e:
            errors += 1
            print(f"ERROR IMAGE: {image_path.name}")

    for mask_path in masks:
        try:
            with Image.open(mask_path) as mask:
                mask.verify()

            with Image.open(mask_path).convert("L") as mask:
                mask_values.update(mask.getdata())

        except Exception:
            errors += 1
            print(f"ERROR MASK: {mask_path.name}")

    print("Image sizes:")
    for size, count in sizes.items():
        print(f"  {size}: {count}")

print("\n" + "=" * 60)
print("GLOBAL RESULTS")
print("=" * 60)

print(f"TOTAL IMAGES : {total_images}")
print(f"TOTAL MASKS  : {total_masks}")
print(f"ERRORS       : {errors}")

print("\nVALIDATION COMPLETE")