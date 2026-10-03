from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.anomaly.statistical import (
    AnomalyDetectionResult,
    StatisticalAnomalyDetector,
)
from src.image_analysis.digital_signature import (
    build_digital_signature_from_image,
)
from src.simulation.runner import run_scenario
from src.simulation.scenarios import SimulationScenario
from src.signature.baseline import StatisticalBaseline
from src.validation.metrics import (
    DetectionMetrics,
    compute_detection_metrics,
)


@dataclass(frozen=True)
class ObservationResult:
    """
    Résultat d'une acquisition individuelle.
    """

    detection: AnomalyDetectionResult

    def validate(self) -> None:
        self.detection.validate()

    @property
    def d2(self) -> float:
        return self.detection.d2

    @property
    def distance(self) -> float:
        return self.detection.distance

    @property
    def state(self) -> str:
        return self.detection.state


@dataclass(frozen=True)
class ScenarioLevelResult:
    """
    Résultat agrégé d'un scénario pour un niveau de sévérité.
    """

    scenario_name: str
    severity: float

    d2_mean: float
    d2_std: float

    distance_mean: float
    distance_std: float

    metrics: DetectionMetrics

    def validate(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                "severity doit être comprise entre 0 et 1."
            )

        values = (
            self.d2_mean,
            self.d2_std,
            self.distance_mean,
            self.distance_std,
        )

        if not all(
            np.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Les statistiques doivent être finies."
            )


@dataclass(frozen=True)
class ScenarioExperimentResult:
    """
    Résultats complets d'un scénario.
    """

    scenario_name: str
    levels: tuple[ScenarioLevelResult, ...]

    def validate(self) -> None:
        if not self.scenario_name:
            raise ValueError(
                "scenario_name ne peut pas être vide."
            )

        if not self.levels:
            raise ValueError(
                "Le scénario doit contenir au moins un niveau."
            )

        for level in self.levels:
            level.validate()

    @property
    def severities(self) -> tuple[float, ...]:
        return tuple(
            level.severity
            for level in self.levels
        )


def build_detector(
    baseline: StatisticalBaseline,
) -> StatisticalAnomalyDetector:
    """
    Construit le détecteur statistique à partir du baseline.
    """

    return StatisticalAnomalyDetector(
        baseline=baseline
    )


def classify_image(
    detector: StatisticalAnomalyDetector,
    image: np.ndarray,
    source: str = "simulated",
) -> ObservationResult:
    """
    Analyse une acquisition via le pipeline officiel :

        image
        ↓
        Digital Signature
        ↓
        Statistical Detector
        ↓
        Detection Result
    """

    signature = build_digital_signature_from_image(
        image,
        source=source,
    )

    detection = detector.detect(
        signature.to_vector()
    )

    result = ObservationResult(
        detection=detection
    )

    result.validate()

    return result


def evaluate_normal_holdout(
    baseline: StatisticalBaseline,
    images: list[np.ndarray]
    | tuple[np.ndarray, ...],
    source: str = "simulated",
) -> DetectionMetrics:
    """
    Évalue des acquisitions supposées normales.
    """

    if not images:
        raise ValueError(
            "Au moins une image holdout est requise."
        )

    detector = build_detector(
        baseline
    )

    distances = []
    states = []

    for image in images:

        result = classify_image(
            detector=detector,
            image=image,
            source=source,
        )

        distances.append(
            result.distance
        )

        states.append(
            result.state
        )

    return compute_detection_metrics(
        distances=distances,
        states=states,
        expected_anomaly=False,
    )


def evaluate_scenario(
    baseline: StatisticalBaseline,
    images: list[np.ndarray]
    | tuple[np.ndarray, ...],
    scenario: SimulationScenario,
    seed: int = 42,
    source: str = "simulated",
) -> ScenarioExperimentResult:
    """
    Exécute un scénario expérimental complet.
    """

    if not images:
        raise ValueError(
            "Au moins une image holdout est requise."
        )

    scenario.validate()

    detector = build_detector(
        baseline
    )

    levels = []

    for severity_index, severity in enumerate(
        scenario.severities
    ):

        d2_values = []
        distance_values = []
        states = []

        single_level = type(scenario)(
            name=scenario.name,
            degradation_type=scenario.degradation_type,
            severities=(severity,),
        )

        for image_index, image in enumerate(
            images
        ):

            controlled_seed = (
                seed
                + severity_index * 1000
                + image_index
            )

            step = run_scenario(
                image=image,
                scenario=single_level,
                seed=controlled_seed,
            )[0]

            result = classify_image(
                detector=detector,
                image=step.result.image,
                source=source,
            )

            d2_values.append(
                result.d2
            )

            distance_values.append(
                result.distance
            )

            states.append(
                result.state
            )

        expected_anomaly = (
            severity > 0.0
        )

        metrics = compute_detection_metrics(
            distances=distance_values,
            states=states,
            expected_anomaly=expected_anomaly,
        )

        level_result = ScenarioLevelResult(
            scenario_name=scenario.name,
            severity=float(severity),
            d2_mean=float(
                np.mean(d2_values)
            ),
            d2_std=float(
                np.std(d2_values)
            ),
            distance_mean=float(
                np.mean(distance_values)
            ),
            distance_std=float(
                np.std(distance_values)
            ),
            metrics=metrics,
        )

        level_result.validate()

        levels.append(
            level_result
        )

    result = ScenarioExperimentResult(
        scenario_name=scenario.name,
        levels=tuple(levels),
    )

    result.validate()

    return result


def evaluate_scenarios(
    baseline: StatisticalBaseline,
    images: list[np.ndarray]
    | tuple[np.ndarray, ...],
    scenarios: list[SimulationScenario]
    | tuple[SimulationScenario, ...],
    seed: int = 42,
    source: str = "simulated",
) -> tuple[ScenarioExperimentResult, ...]:
    """
    Évalue plusieurs scénarios avec le même détecteur statistique.
    """

    if not scenarios:
        raise ValueError(
            "Au moins un scénario est requis."
        )

    return tuple(
        evaluate_scenario(
            baseline=baseline,
            images=images,
            scenario=scenario,
            seed=seed,
            source=source,
        )
        for scenario in scenarios
    )