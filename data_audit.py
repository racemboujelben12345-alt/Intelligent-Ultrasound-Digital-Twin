from pathlib import Path
from collections import Counter

from PIL import Image


# ============================================================
# SCAN A DIGITAL TWIN V2
# DATA AUDIT - PUBLIC ULTRASOUND DATASET
# ============================================================

DATASET_ROOT = Path(
    "data/raw/public_ultrasound/us_simulation/abdominal_US/AUS/images"
)


def audit_split(split_name: str):
    """
    Analyse un split du dataset : train ou test.
    """

    split_dir = DATASET_ROOT / split_name

    print("\n" + "=" * 70)
    print(f"DATA AUDIT : {split_name.upper()}")
    print("=" * 70)

    if not split_dir.exists():
        print(f"[ERROR] Dossier introuvable : {split_dir}")
        return

    # Recherche récursive des fichiers PNG
    image_files = list(split_dir.rglob("*.png"))

    print(f"Nombre de fichiers PNG : {len(image_files)}")

    if not image_files:
        print("[WARNING] Aucune image PNG trouvée.")
        return

    dimensions = Counter()
    modes = Counter()

    valid_images = 0
    invalid_images = 0

    for image_path in image_files:

        try:
            with Image.open(image_path) as img:

                # Vérification réelle du fichier image
                img.verify()

            # Réouverture après verify()
            with Image.open(image_path) as img:

                dimensions[img.size] += 1
                modes[img.mode] += 1

            valid_images += 1

        except Exception as error:
            invalid_images += 1

            print(
                f"[INVALID] {image_path.name} -> {error}"
            )

    print("\n--- VALIDATION ---")
    print(f"Images valides   : {valid_images}")
    print(f"Images invalides : {invalid_images}")

    print("\n--- DIMENSIONS ---")

    for dimension, count in dimensions.most_common():
        print(
            f"{dimension[0]} x {dimension[1]} : {count} images"
        )

    print("\n--- MODES ---")

    for mode, count in modes.most_common():
        print(
            f"{mode} : {count} images"
        )


def main():
    print("=" * 70)
    print("SCAN A DIGITAL TWIN V2")
    print("PUBLIC ULTRASOUND DATA AUDIT")
    print("=" * 70)

    audit_split("train")
    audit_split("test")

    print("\n" + "=" * 70)
    print("AUDIT TERMINÉ")
    print("=" * 70)


if __name__ == "__main__":
    main()