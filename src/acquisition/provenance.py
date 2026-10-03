from dataclasses import dataclass
from enum import Enum


class DataSource(str, Enum):
    """
    Origine contrôlée d'une acquisition.
    """

    SCAN_A = "scan_a"
    PUBLIC = "public"
    SIMULATED = "simulated"


@dataclass(frozen=True)
class DataProvenance:
    """
    Métadonnées minimales permettant de tracer
    l'origine d'une donnée.
    """

    source: DataSource
    dataset: str
    relative_path: str
    is_real_scan_a: bool
    is_public_reference: bool
    is_simulation: bool

    def validate(self) -> None:
        """
        Vérifie la cohérence de la provenance.
        """

        flags = [
            self.is_real_scan_a,
            self.is_public_reference,
            self.is_simulation,
        ]

        if sum(flags) != 1:
            raise ValueError(
                "Une acquisition doit appartenir à une seule "
                "catégorie : SCAN A, PUBLIC ou SIMULATION."
            )

        if self.source == DataSource.SCAN_A and not self.is_real_scan_a:
            raise ValueError("Incohérence : source SCAN A.")

        if self.source == DataSource.PUBLIC and not self.is_public_reference:
            raise ValueError("Incohérence : source PUBLIC.")

        if self.source == DataSource.SIMULATED and not self.is_simulation:
            raise ValueError("Incohérence : source SIMULATED.")