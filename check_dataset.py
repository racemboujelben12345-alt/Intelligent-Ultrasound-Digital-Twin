from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

ROOT = Path("data/raw/public_ultrasound/us_simulation/abdominal_US/AUS")

sizes = Counter()
mask_values = Counter()

total = 0
bad = 0
size_mismatch = 0
empty_masks = 0

for split in ["train", "test"]:
    image_dir = ROOT / "images" / split
    annotation_dir = ROOT / "annotations" / split

    images = sorted(image_dir.iterdir())

    print(f"\n{split.upper()}")
    print("-" * 40)
    print("Images:", len(images))

    for image_path in images:
        annotation_path = annotation_dir / image_path.name
        total += 1

        try:
            image = Image.open(image_path)
            annotation = Image.open(annotation_path)

            sizes[image.size] += 1

            if image.size != annotation.size:
                size_mismatch += 1

            mask = np.array(annotation)

            mask_values.update(np.unique(mask).tolist())

            if not np.any(mask):
                empty_masks += 1

        except Exception as e:
            bad += 1
            print("Problem:", image_path.name, "->", e)

print("\n" + "=" * 50)
print("USSimAndSegm DATASET VALIDATION")
print("=" * 50)

print("TOTAL PAIRS       :", total)
print("CORRUPTED PAIRS   :", bad)
print("SIZE MISMATCH     :", size_mismatch)
print("EMPTY MASKS       :", empty_masks)

print("\nIMAGE SIZES:")
for size, count in sizes.most_common():
    print(f"  {size}: {count}")

print("\nMASK VALUES:")
print(sorted(mask_values.items()))

print("\nVALIDATION COMPLETE")