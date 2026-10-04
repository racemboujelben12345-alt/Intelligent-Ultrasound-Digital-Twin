from pathlib import Path
from PIL import Image

ROOT = Path("data/raw/public_ultrasound/busi/Dataset_BUSI_with_GT")

classes = ["benign", "malignant", "normal"]

total_images = 0
matched_images = 0
missing_masks = 0
corrupted = 0

print("=" * 70)
print("BUSI IMAGE-MASK CORRESPONDENCE VALIDATION")
print("=" * 70)

for cls in classes:

    folder = ROOT / cls

    image_files = [
        p for p in folder.glob("*.png")
        if "_mask" not in p.stem
    ]

    print(f"\n{cls.upper()}")
    print("-" * 50)

    for image_path in image_files:

        total_images += 1

        base = image_path.stem

        # All masks associated with this image
        masks = list(folder.glob(f"{base}_mask*.png"))

        if not masks:
            print(f"MISSING MASK : {image_path.name}")
            missing_masks += 1
            continue

        try:
            with Image.open(image_path) as img:
                image_size = img.size
                img.load()

            valid_masks = 0

            for mask_path in masks:

                with Image.open(mask_path) as mask:
                    mask_size = mask.size
                    mask.load()

                if mask_size != image_size:
                    print(
                        f"SIZE MISMATCH : "
                        f"{image_path.name} <-> {mask_path.name}"
                    )
                else:
                    valid_masks += 1

            if valid_masks > 0:
                matched_images += 1

        except Exception as e:
            print(f"CORRUPTED : {image_path.name} -> {e}")
            corrupted += 1

print("\n" + "=" * 70)
print("GLOBAL RESULTS")
print("=" * 70)

print(f"TOTAL IMAGES       : {total_images}")
print(f"MATCHED IMAGES     : {matched_images}")
print(f"MISSING MASKS      : {missing_masks}")
print(f"CORRUPTED IMAGES   : {corrupted}")

print("\nVALIDATION COMPLETE")
