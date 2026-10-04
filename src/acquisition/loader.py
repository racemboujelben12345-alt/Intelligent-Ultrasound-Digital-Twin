"""
Intelligent Ultrasound Digital Twin V2
======================

Acquisition Loader
------------------

Rôle
----
Charger les acquisitions expérimentales ou générer
des données de démonstration contrôlées.

Sources supportées
------------------
- acquisition expérimentale réelle
- données publiques de référence
- données simulées

Important
---------
Ce module ne :
- calcule pas les features ;
- construit pas le Digital Signature ;
- détecte pas les anomalies ;
- ne réalise pas le monitoring temporel.

La provenance des données est explicitement conservée.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import (
    ANALYSIS_SIZE,
    DEMO_DATA_DIR,
    EXPERIMENTAL_DATA_DIR,
)
from src.acquisition.metadata import load_metadata
from src.acquisition.models import Acquisition
from src.acquisition.provenance import (
    DataProvenance,
    DataSource,
)
from src.simulation.degradation import (
    DegradationType,
    apply_degradation,
)


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def _load_image(
    path: Path,
    *,
    resize_to: int | None = ANALYSIS_SIZE,
) -> np.ndarray:
    """
    Charge une image grayscale et retourne une image float32 [0,1].
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Image introuvable : {path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"Impossible de lire l'image : {path}"
        )

    image = image.astype(
        np.float32
    ) / 255.0

    if resize_to is not None:
        image = cv2.resize(
            image,
            (resize_to, resize_to),
            interpolation=cv2.INTER_AREA,
        )

    return np.clip(
        image,
        0.0,
        1.0,
    ).astype(np.float32)


def _phantom(
    *,
    size: int = ANALYSIS_SIZE,
    seed: int = 42,
) -> np.ndarray:
    """
    Génère un phantom ultrasonore synthétique de démonstration.

    Cette image est exclusivement destinée aux tests.
    Elle ne représente pas une acquisition acquisition expérimentale réellele.
    """

    rng = np.random.default_rng(
        seed
    )

    image = np.zeros(
        (size, size),
        dtype=np.float32,
    )

    yy, xx = np.mgrid[
        0:size,
        0:size,
    ]

    cx = size / 2.0
    cy = size / 2.0

    # Background soft tissue-like field
    radial = np.sqrt(
        ((xx - cx) / size) ** 2
        + ((yy - cy) / size) ** 2
    )

    background = np.exp(
        -5.0 * radial
    )

    image += (
        0.28
        + 0.20 * background
    )

    # Synthetic structures
    structures = [
        (0.32, 0.35, 0.10, 0.12, 0.22),
        (0.68, 0.36, 0.09, 0.14, -0.16),
        (0.50, 0.67, 0.15, 0.08, 0.18),
    ]

    for nx, ny, rx, ry, amplitude in structures:

        x0 = nx * size
        y0 = ny * size

        ellipse = (
            ((xx - x0) / (rx * size)) ** 2
            + ((yy - y0) / (ry * size)) ** 2
        )

        image += amplitude * np.exp(
            -3.0 * ellipse
        )

    # Low-frequency texture
    texture = rng.normal(
        0.0,
        1.0,
        (size, size),
    )

    texture = cv2.GaussianBlur(
        texture.astype(np.float32),
        (0, 0),
        sigmaX=2.0,
    )

    texture_std = float(
        np.std(texture)
    )

    if texture_std > 1e-12:
        texture /= texture_std

    image += (
        0.025
        * texture
    )

    # Mild multiplicative speckle
    speckle = rng.normal(
        0.0,
        0.05,
        (size, size),
    )

    image *= (
        1.0 + speckle
    )

    return np.clip(
        image,
        0.0,
        1.0,
    ).astype(np.float32)


def generate_demo_acquisitions(
    *,
    n_samples: int = 50,
    seed: int = 42,
) -> tuple[Acquisition, ...]:
    """
    Génère une population contrôlée d'acquisitions simulées.

    Les acquisitions sont indépendantes et portent une provenance
    explicite SIMULATED.
    """

    if n_samples < 2:
        raise ValueError(
            "n_samples doit être >= 2."
        )

    DEMO_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    acquisitions = []

    for index in range(
        n_samples
    ):

        image = _phantom(
            size=ANALYSIS_SIZE,
            seed=seed + index,
        )

        # Small normal operating variation
        rng = np.random.default_rng(
            seed + 1000 + index
        )

        normal_noise = rng.normal(
            0.0,
            0.008,
            image.shape,
        ).astype(np.float32)

        image = np.clip(
            image + normal_noise,
            0.0,
            1.0,
        ).astype(np.float32)

        path = (
            DEMO_DATA_DIR
            / f"demo_{index:04d}.png"
        )

        cv2.imwrite(
            str(path),
            (
                image * 255.0
            ).astype(np.uint8),
        )

        provenance = DataProvenance(
            source=DataSource.SIMULATED,
            dataset="SCAN_A_Digital_Twin_Demo",
            relative_path=str(
                path.relative_to(
                    DEMO_DATA_DIR.parent.parent
                )
            ),
            is_real_scan_a=False,
            is_public_reference=False,
            is_simulation=True,
        )

        acquisition = Acquisition(
            id=f"demo_{index:04d}",
            source=DataSource.SIMULATED.value,
            t=index,
            image=image,
            provenance=provenance,
            truth_level=0.0,
            simulation_scenario="normal_reference",
            simulation_severity=0.0,
        )

        acquisition.validate()

        acquisitions.append(
            acquisition
        )

    return tuple(
        acquisitions
    )


def load_experimental_acquisitions(
    directory: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Acquisition, ...]:
    """
    Charge les acquisitions réelles disponibles dans EXPERIMENTAL_DATA_DIR.

    Aucun paramètre d'acquisition n'est inventé.
    Les métadonnées absentes restent absentes.
    """

    root = (
        EXPERIMENTAL_DATA_DIR
        if directory is None
        else Path(directory)
    )

    if not root.exists():
        return tuple()

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    metadata = load_metadata(metadata_path)
    metadata_by_file = {
        str(row.get("file_path", "")).replace("\\", "/").lstrip("./"): row
        for row in metadata.values()
        if row.get("file_path")
    }

    acquisitions = []

    for index, path in enumerate(
        paths
    ):

        image = _load_image(
            path
        )

        provenance = DataProvenance(
            source=DataSource.SCAN_A,
            dataset="SCAN_A",
            relative_path=str(
                path.relative_to(root)
            ),
            is_real_scan_a=True,
            is_public_reference=False,
            is_simulation=False,
        )

        relative_path = str(path.relative_to(root)).replace("\\\\", "/")
        fallback_id = f"scan_a_{index:04d}"
        row = metadata_by_file.get(relative_path, metadata.get(fallback_id, {}))
        acquisition_id = row.get("acquisition_id", fallback_id)

        session_id = row.get("session_id")
        timestamp = row.get("timestamp")

        params = {
            key: row[key]
            for key in (
                "campaign_phase",
                "device_id",
                "probe_id",
                "preset",
                "frequency",
                "gain",
                "depth",
                "focus",
                "target_id",
                "operator_id",
                "file_format",
                "tgc",
                "dynamic_range",
                "dimensions",
                "bit_depth",
                "frame_count",
                "quality_flag",
                "notes",
            )
            if key in row
        }

        acquisition = Acquisition(
            id=acquisition_id,
            source=DataSource.SCAN_A.value,
            t=index,
            image=image,
            params=params,
            provenance=provenance,
            session_id=session_id,
            timestamp=timestamp,
            truth_level=0.0,
        )

        acquisition.validate()

        acquisitions.append(
            acquisition
        )

    return tuple(
        acquisitions
    )


def build_controlled_degradation(
    acquisition: Acquisition,
    degradation_type: DegradationType | str,
    severity: float,
    *,
    seed: int = 42,
) -> Acquisition:
    """
    Applique une dégradation numérique à une acquisition.

    L'acquisition originale n'est jamais modifiée.
    """

    if not isinstance(
        acquisition,
        Acquisition,
    ):
        raise TypeError(
            "acquisition doit être une Acquisition."
        )

    result = apply_degradation(
        image=acquisition.image,
        degradation_type=degradation_type,
        severity=severity,
        seed=seed,
    )

    provenance = DataProvenance(
        source=DataSource.SIMULATED,
        dataset="SCAN_A_Digital_Twin_Simulation",
        relative_path=(
            acquisition.provenance.relative_path
            if acquisition.provenance is not None
            else acquisition.id
        ),
        is_real_scan_a=False,
        is_public_reference=False,
        is_simulation=True,
    )

    degraded = Acquisition(
        id=(
            f"{acquisition.id}_"
            f"{str(result.degradation_type)}_"
            f"{severity:.2f}"
        ),
        source=DataSource.SIMULATED.value,
        t=acquisition.t,
        image=result.image,
        params=dict(
            acquisition.params
        ),
        provenance=provenance,
        session_id=acquisition.session_id,
        timestamp=acquisition.timestamp,
        truth_level=float(
            severity
        ),
        simulation_scenario=(
            result.degradation_type
        ),
        simulation_severity=float(
            severity
        ),
    )

    degraded.validate()

    return degraded


def load_acquisitions(
    *,
    prefer_experimental: bool = True,
    minimum_experimental: int = 30,
    prefer_scan_a: bool | None = None,
    minimum_scan_a: int | None = None,
    demo_size: int = 50,
    seed: int = 42,
) -> tuple[Acquisition, ...]:
    """
    Point d'entrée principal de la couche acquisition.

    Priorité :
        1. acquisition expérimentale réelle si suffisamment d'acquisitions existent ;
        2. données simulées de démonstration sinon.

    Le fallback simulé est explicitement marqué SIMULATED.
    """

    if prefer_scan_a is not None:
        prefer_experimental = prefer_scan_a
    if minimum_scan_a is not None:
        minimum_experimental = minimum_scan_a

    if prefer_experimental:
        experimental = load_experimental_acquisitions()

        if len(experimental) >= minimum_experimental:
            return experimental

    return generate_demo_acquisitions(
        n_samples=demo_size,
        seed=seed,
    )


def get_acquisitions(
    *,
    prefer_experimental: bool = True,
    minimum_experimental: int = 30,
    prefer_scan_a: bool | None = None,
    minimum_scan_a: int | None = None,
    demo_size: int = 50,
    seed: int = 42,
) -> tuple[Acquisition, ...]:
    """
    Alias explicite du point d'entrée acquisition V2.
    """

    return load_acquisitions(
        prefer_experimental=prefer_experimental,
        minimum_experimental=minimum_experimental,
        prefer_scan_a=prefer_scan_a,
        minimum_scan_a=minimum_scan_a,
        demo_size=demo_size,
        seed=seed,
    )

# Backward-compatible alias for legacy callers.
load_scan_a_acquisitions = load_experimental_acquisitions
