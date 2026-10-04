from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DataSource(str, Enum):
    """Origine contrôlée d'une acquisition."""

    EXPERIMENTAL = "experimental"
    # Legacy value kept for backward compatibility with older SCAN A data.
    SCAN_A = "scan_a"
    PUBLIC = "public"
    SIMULATED = "simulated"


@dataclass(frozen=True)
class DataProvenance:
    """
    Traçabilité de l'origine d'une donnée.

    Les catégories sont volontairement exclusives:
    experimental, public ou simulated.
    """

    source: DataSource
    dataset: str
    relative_path: str
    is_experimental: bool = False
    is_public_reference: bool = False
    is_simulation: bool = False
    # Legacy flag retained so old serialized metadata remains readable.
    is_real_scan_a: bool = False

    def validate(self) -> None:
        flags = [
            self.is_experimental,
            self.is_public_reference,
            self.is_simulation,
        ]

        # Legacy SCAN A provenance maps to the experimental category.
        if self.source == DataSource.SCAN_A and self.is_real_scan_a:
            flags = [True, False, False]

        if sum(flags) != 1:
            raise ValueError(
                "Une acquisition doit appartenir à une seule catégorie : "
                "EXPERIMENTAL, PUBLIC ou SIMULATED."
            )

        if self.source == DataSource.EXPERIMENTAL and not self.is_experimental:
            raise ValueError("Incohérence : source EXPERIMENTAL.")

        if self.source == DataSource.SCAN_A and not self.is_real_scan_a and not self.is_experimental:
            raise ValueError("Incohérence : source SCAN_A.")

        if self.source == DataSource.PUBLIC and not self.is_public_reference:
            raise ValueError("Incohérence : source PUBLIC.")

        if self.source == DataSource.SIMULATED and not self.is_simulation:
            raise ValueError("Incohérence : source SIMULATED.")

    @property
    def source_category(self) -> str:
        """Catégorie canonique utilisée par les métriques et validations."""
        if self.source in (DataSource.EXPERIMENTAL, DataSource.SCAN_A):
            return "experimental"
        return self.source.value
