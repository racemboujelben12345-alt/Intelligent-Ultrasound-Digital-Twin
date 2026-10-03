from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


IMAGE_PATH = Path(
    "data/raw/public_ultrasound/us_simulation/abdominal_US/AUS/images/train/ct14-1.png"
)


def main():
    if not IMAGE_PATH.exists():
        print(f"[ERROR] Image introuvable : {IMAGE_PATH}")
        return

    with Image.open(IMAGE_PATH) as img:
        print("=" * 70)
        print("SCAN A DIGITAL TWIN V2")
        print("PUBLIC ULTRASOUND IMAGE INSPECTION")
        print("=" * 70)

        print(f"\nFichier      : {IMAGE_PATH.name}")
        print(f"Dimensions   : {img.size}")
        print(f"Mode         : {img.mode}")
        print(f"Format       : {img.format}")

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Ultrasound reference - {IMAGE_PATH.name}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()