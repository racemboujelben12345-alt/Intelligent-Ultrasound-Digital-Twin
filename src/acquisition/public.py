from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.acquisition.models import Acquisition
from src.acquisition.provenance import (
    DataProvenance,
    DataSource,
)


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


EXCLUDED_TOKENS = (
    "mask",
    "masks",
    "annotation",
    "annotations",
    "segmentation",
    "segmentations",
    "ground_truth",
    "groundtruth",
)


def _is_excluded(path: Path) -> bool:
    """
    Évite de charger des masques ou annotations comme des images.
    """
    parts = {
        part.lower()
        for part in path.parts
    }

    return any(
        token in part
        for part in parts
        for token in EXCLUDED_TOKENS
    )


def _normalize_image(
    image: np.ndarray,
    resize_to: int = 128,
) -> np.ndarray:
    """
    Convertit une image en niveau de gris float32 [0, 1]
    avec une taille d'analyse fixe.
    """

    if image is None:
        raise ValueError("Image vide ou illisible.")

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    if image.ndim != 2:
        raise ValueError(
            f"Dimensions d'image non supportées : {image.shape}"
        )

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0

    else:
        image = image.astype(np.float32)

        min_value = float(np.min(image))
        max_value = float(np.max(image))

        if min_value < 0.0 or max_value > 1.0:
            if max_value > min_value:
                image = (
                    image - min_value
                ) / (
                    max_value - min_value
                )
            else:
                image = np.zeros_like(
                    image,
                    dtype=np.float32,
                )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image = cv2.resize(
        image,
        (resize_to, resize_to),
        interpolation=cv2.INTER_AREA,
    )

    return image.astype(np.float32)


def load_public_reference_acquisitions(
    directory: str | Path,
    *,
    resize_to: int = 128,
    limit: int | None = None,
) -> list[Acquisition]:
    """
    Charge des images publiques comme données de référence.

    IMPORTANT :
    ces images sont explicitement marquées PUBLIC_REFERENCE.
    Elles ne peuvent pas être utilisées comme baseline SCAN A.

    Parameters
    ----------
    directory:
        Répertoire racine du dataset public.

    resize_to:
        Taille de l'image utilisée par le pipeline V2.

    limit:
        Nombre maximal d'images à charger.
    """

    root = Path(directory)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset public introuvable : {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Le chemin n'est pas un dossier : {root}"
        )

    image_paths = sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
            and not _is_excluded(path)
        )
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "limit doit être strictement positif."
            )

        image_paths = image_paths[:limit]

    acquisitions: list[Acquisition] = []

    for index, path in enumerate(image_paths):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            continue

        try:
            normalized = _normalize_image(
                image,
                resize_to=resize_to,
            )
        except ValueError:
            continue

        try:
            relative_path = str(
                path.relative_to(root)
            )
        except ValueError:
            relative_path = str(path)

        provenance = DataProvenance(
            source=DataSource.PUBLIC,
            dataset=root.name,
            relative_path=relative_path,
            is_real_scan_a=False,
            is_public_reference=True,
            is_simulation=False,
        )

        provenance.validate()

        acquisition = Acquisition(
            id=f"public_{index:05d}",
            source=DataSource.PUBLIC.value,
            t=index,
            image=normalized,
            params={
                "path": relative_path,
                "dataset": root.name,
            },
            provenance=provenance,
            session_id="public_reference",
            timestamp=None,
            truth_level=0.0,
            simulation_scenario=None,
            simulation_severity=None,
        )

        acquisitions.append(
            acquisition
        )

    if not acquisitions:
        raise RuntimeError(
            f"Aucune image publique exploitable trouvée dans : {root}"
        )

    return acquisitions


def load_public(
    directory: str | Path,
    *,
    resize_to: int = 128,
    limit: int | None = None,
) -> list[Acquisition]:
    """
    Alias explicite pour le benchmark public V2.
    """
    return load_public_reference_acquisitions(
        directory,
        resize_to=resize_to,
        limit=limit,
    )