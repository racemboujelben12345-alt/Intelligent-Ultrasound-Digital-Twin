from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """
    Séparation déterministe d'une population d'acquisitions.

    training:
        Utilisé exclusivement pour construire la baseline.

    calibration:
        Utilisé exclusivement pour calibrer le monitoring temporel.

    test:
        Réservé à l'évaluation finale.
    """

    training: tuple[np.ndarray, ...]
    calibration: tuple[np.ndarray, ...]
    test: tuple[np.ndarray, ...]

    def validate(self) -> None:
        if len(self.training) < 2:
            raise ValueError(
                "Le training set doit contenir au moins 2 images."
            )

        if len(self.calibration) < 2:
            raise ValueError(
                "Le calibration set doit contenir au moins 2 images."
            )

        if len(self.test) < 1:
            raise ValueError(
                "Le test set doit contenir au moins 1 image."
            )

        for name, images in (
            ("training", self.training),
            ("calibration", self.calibration),
            ("test", self.test),
        ):
            for image in images:

                array = np.asarray(image)

                if array.ndim != 2:
                    raise ValueError(
                        f"{name} contient une image qui n'est pas 2D."
                    )

                if not np.all(
                    np.isfinite(array)
                ):
                    raise ValueError(
                        f"{name} contient une image non finie."
                    )

    @property
    def total_size(self) -> int:
        return (
            len(self.training)
            + len(self.calibration)
            + len(self.test)
        )


def split_dataset(
    images: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    training_size: int,
    calibration_size: int,
    test_size: int,
) -> DatasetSplit:
    """
    Effectue une séparation déterministe sans mélange.

    L'ordre des acquisitions est conservé.

    Parameters
    ----------
    images:
        Population complète.

    training_size:
        Nombre d'acquisitions pour la baseline.

    calibration_size:
        Nombre d'acquisitions pour le monitoring temporel.

    test_size:
        Nombre d'acquisitions réservées au test final.
    """

    population = tuple(
        np.asarray(image)
        for image in images
    )

    total_required = (
        training_size
        + calibration_size
        + test_size
    )

    if len(population) != total_required:
        raise ValueError(
            "La taille de la population ne correspond pas "
            "à la somme des tailles demandées : "
            f"{len(population)} != {total_required}."
        )

    if training_size < 2:
        raise ValueError(
            "training_size doit être >= 2."
        )

    if calibration_size < 2:
        raise ValueError(
            "calibration_size doit être >= 2."
        )

    if test_size < 1:
        raise ValueError(
            "test_size doit être >= 1."
        )

    result = DatasetSplit(
        training=population[
            :training_size
        ],
        calibration=population[
            training_size:
            training_size + calibration_size
        ],
        test=population[
            training_size + calibration_size:
        ],
    )

    result.validate()

    return result