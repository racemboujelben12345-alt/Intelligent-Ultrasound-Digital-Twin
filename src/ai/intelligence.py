"""Expert AI intelligence layer for the Intelligent Ultrasound Digital Twin.

The module deliberately separates:
- unsupervised anomaly evidence;
- supervised engineering-state classification;
- predictive regression;
- model metadata / provenance;
- feature-level anomaly explainability.

Scientific boundary
-------------------
AI outputs are engineering evidence. They are not calibrated physical-failure
probabilities, clinical diagnoses, or proof of hardware failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AIModelCard:
    """Auditable description of one fitted AI configuration."""

    model_version: str
    task: str
    feature_names: tuple[str, ...]
    training_samples: int
    training_source: str
    random_seeds: tuple[int, ...]
    hyperparameters: tuple[tuple[str, str], ...]
    fitted_at_utc: str
    input_contract_hash: str

    def validate(self) -> None:
        if not self.model_version or not self.task:
            raise ValueError("Model card requires model_version and task.")
        if self.training_samples < 2:
            raise ValueError("training_samples must be >= 2.")
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty.")
        if not self.input_contract_hash:
            raise ValueError("input_contract_hash cannot be empty.")


@dataclass(frozen=True)
class AIInferenceRecord:
    """Traceable record of one AI inference."""

    acquisition_id: str
    model_version: str
    task: str
    input_hash: str
    source: str
    output: tuple[tuple[str, float], ...]
    confidence: float
    created_at_utc: str

    def validate(self) -> None:
        if not self.acquisition_id:
            raise ValueError("acquisition_id cannot be empty.")
        if not self.input_hash or not self.model_version:
            raise ValueError("Inference provenance is incomplete.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")


@dataclass(frozen=True)
class AIAnomalyAssessment:
    anomaly_score: float
    anomaly_vote_rate: float
    confidence: float
    ensemble_agreement: float
    model_scores: tuple[float, ...]
    state: str

    @property
    def anomaly_probability(self) -> float:
        """Backward-compatible alias; this is a vote fraction, not a calibrated probability."""
        return self.anomaly_vote_rate

    def validate(self) -> None:
        values = (self.anomaly_score, self.anomaly_vote_rate,
                  self.confidence, self.ensemble_agreement)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("AI assessment contains non-finite values.")
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError("AI scores must be in [0, 1].")
        if self.state not in {"NOMINAL", "WATCH", "ANOMALY"}:
            raise ValueError("Invalid AI state.")


@dataclass(frozen=True)
class AIPrediction:
    """Prediction with an empirical ensemble-spread interval, not calibrated coverage."""
    prediction: float
    lower: float
    upper: float
    confidence: float
    interval_method: str = "ensemble_empirical_p05_p95"

    def validate(self) -> None:
        if not all(np.isfinite(v) for v in
                   (self.prediction, self.lower, self.upper, self.confidence)):
            raise ValueError("Prediction contains non-finite values.")
        if self.lower > self.prediction or self.prediction > self.upper:
            raise ValueError("Invalid prediction interval.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        if self.interval_method != "ensemble_empirical_p05_p95":
            raise ValueError("Unsupported interval_method.")


def _hash_array(x: np.ndarray) -> str:
    """Stable SHA-256 hash for an inference input vector."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _contract_hash(feature_names: tuple[str, ...]) -> str:
    return hashlib.sha256("|".join(feature_names).encode("utf-8")).hexdigest()


class UltrasoundAIEngine:
    """Multi-task, provenance-aware AI engine.

    Each task owns its scaler and fitted models, preventing accidental model
    contamination when several AI modes are trained in the same process.
    """

    MODEL_VERSION = "ai-ultrasound-ensemble-v2.0"

    def __init__(
        self,
        *,
        n_trees: int = 200,
        n_models: int = 7,
        random_seeds: tuple[int, ...] = (11, 23, 37, 41, 53, 67, 79),
        contamination: float | str = "auto",
    ) -> None:
        if n_trees < 50:
            raise ValueError("n_trees must be >= 50.")
        if n_models < 3:
            raise ValueError("n_models must be >= 3.")
        if len(random_seeds) != n_models:
            raise ValueError("random_seeds length must equal n_models.")
        self.n_trees = int(n_trees)
        self.n_models = int(n_models)
        self.random_seeds = tuple(random_seeds)
        self.contamination = contamination

        self.unsupervised_scaler = StandardScaler()
        self.supervised_scaler = StandardScaler()
        self.predictive_scaler = StandardScaler()

        self.models: list[IsolationForest] = []
        self.supervised_model: RandomForestClassifier | None = None
        self.regressors: list[RandomForestRegressor] = []
        self.feature_names: tuple[str, ...] = ()
        self.reference_mean: np.ndarray | None = None
        self.reference_anomaly_scores: np.ndarray | None = None
        self.model_cards: dict[str, AIModelCard] = {}
        self.fitted = False

    @staticmethod
    def _matrix(X: np.ndarray, *, min_rows: int = 2) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[0] < min_rows:
            raise ValueError(f"X must be a 2D matrix with >= {min_rows} rows.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf.")
        return X

    def _set_features(self, n_features: int, names: tuple[str, ...] | None) -> None:
        resolved = tuple(names or [f"feature_{i}" for i in range(n_features)])
        if len(resolved) != n_features or len(set(resolved)) != n_features:
            raise ValueError("feature_names must be unique and match X.")
        self.feature_names = resolved

    def _card(self, task: str, n_samples: int, source: str, params: dict[str, Any]) -> AIModelCard:
        card = AIModelCard(
            model_version=self.MODEL_VERSION,
            task=task,
            feature_names=self.feature_names,
            training_samples=n_samples,
            training_source=source,
            random_seeds=self.random_seeds,
            hyperparameters=tuple(sorted((k, str(v)) for k, v in params.items())),
            fitted_at_utc=datetime.now(timezone.utc).isoformat(),
            input_contract_hash=_contract_hash(self.feature_names),
        )
        card.validate()
        self.model_cards[task] = card
        return card

    def fit_reference(
        self,
        X_reference: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
        training_source: str = "reference",
    ) -> "UltrasoundAIEngine":
        X = self._matrix(X_reference)
        self._set_features(X.shape[1], feature_names)
        self.reference_mean = np.mean(X, axis=0)
        Xs = self.unsupervised_scaler.fit_transform(X)

        self.models = []
        for seed in self.random_seeds:
            model = IsolationForest(
                n_estimators=self.n_trees,
                contamination=self.contamination,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(Xs)
            self.models.append(model)

        # Calibrate anomaly evidence against the reference population itself.
        # This is an empirical percentile, not a calibrated probability.
        reference_raw = np.vstack([
            -model.score_samples(Xs) for model in self.models
        ])
        self.reference_anomaly_scores = np.mean(reference_raw, axis=0)

        self._card("unsupervised_anomaly", len(X), training_source, {
            "n_trees": self.n_trees, "n_models": self.n_models,
            "contamination": self.contamination,
        })
        self.fitted = True
        return self

    def assess(self, x: np.ndarray) -> AIAnomalyAssessment:
        if not self.fitted or not self.models:
            raise RuntimeError("Call fit_reference() before assess().")
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != len(self.feature_names) or not np.all(np.isfinite(x)):
            raise ValueError("x is incompatible with the AI feature contract.")

        xs = self.unsupervised_scaler.transform(x)
        raw = np.asarray([-float(m.score_samples(xs)[0]) for m in self.models])
        center = float(np.mean(raw))
        if self.reference_anomaly_scores is None:
            raise RuntimeError("Reference anomaly calibration is unavailable.")

        # Empirical CDF / percentile rank of the new sample in the reference
        # population. A high percentile means stronger anomaly evidence.
        score = float(
            np.mean(self.reference_anomaly_scores <= center)
        )

        votes = np.asarray([m.predict(xs)[0] == -1 for m in self.models])
        vote_rate = float(np.mean(votes))
        agreement = float(max(vote_rate, 1.0 - vote_rate))
        dispersion = float(np.std(raw) / (np.mean(np.abs(raw)) + 1e-9))
        confidence = float(np.clip(0.5 * agreement + 0.5 * (1.0 - dispersion), 0.0, 1.0))
        state = ("ANOMALY" if vote_rate >= 0.67 or score >= 0.80
                 else "WATCH" if vote_rate >= 0.34 or score >= 0.50
                 else "NOMINAL")

        result = AIAnomalyAssessment(score, vote_rate, confidence, agreement,
                                     tuple(float(v) for v in raw), state)
        result.validate()
        return result

    def anomaly_feature_contributions(self, x: np.ndarray) -> list[tuple[str, float]]:
        """Estimate feature influence by leave-one-feature-out perturbation.

        Each feature is replaced by its reference mean and the change in
        ensemble anomaly evidence is measured. This is a sensitivity measure,
        not causal attribution.
        """
        if self.reference_mean is None:
            raise RuntimeError("Call fit_reference() first.")
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(x) != len(self.feature_names):
            raise ValueError("x is incompatible with the AI feature contract.")
        base = self.assess(x).anomaly_score
        contributions = []
        for i, name in enumerate(self.feature_names):
            perturbed = x.copy()
            perturbed[i] = self.reference_mean[i]
            delta = max(0.0, base - self.assess(perturbed).anomaly_score)
            contributions.append((name, float(delta)))
        total = sum(v for _, v in contributions)
        if total > 0:
            contributions = [(n, v / total) for n, v in contributions]
        return sorted(contributions, key=lambda item: item[1], reverse=True)

    def fit_supervised(self, X: np.ndarray, y: np.ndarray,
                       training_source: str = "validated_labels") -> "UltrasoundAIEngine":
        X = self._matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1D array aligned with X.")
        if len(np.unique(y)) < 2:
            raise ValueError("Supervised AI requires at least two classes.")
        self._set_features(X.shape[1], self.feature_names if self.feature_names else None)
        Xs = self.supervised_scaler.fit_transform(X)
        self.supervised_model = RandomForestClassifier(
            n_estimators=self.n_trees, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )
        self.supervised_model.fit(Xs, y)
        self._card("supervised_classifier", len(X), training_source,
                   {"n_trees": self.n_trees, "class_weight": "balanced"})
        return self

    def predict_class(self, x: np.ndarray) -> dict:
        if self.supervised_model is None:
            raise RuntimeError("Call fit_supervised() first.")
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != len(self.feature_names) or not np.all(np.isfinite(x)):
            raise ValueError("x is incompatible with the AI feature contract.")
        probabilities = self.supervised_model.predict_proba(
            self.supervised_scaler.transform(x)
        )[0]
        index = int(np.argmax(probabilities))
        return {
            "class": self.supervised_model.classes_[index].item(),
            "probability": float(probabilities[index]),
            "probabilities": {str(k): float(v) for k, v in
                              zip(self.supervised_model.classes_, probabilities)},
            "feature_importance": {n: float(v) for n, v in
                                   zip(self.feature_names, self.supervised_model.feature_importances_)},
        }

    def fit_predictive_ensemble(self, X_history: np.ndarray, y_history: np.ndarray,
                                training_source: str = "temporal_history") -> "UltrasoundAIEngine":
        X = self._matrix(X_history)
        y = np.asarray(y_history, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(X) or not np.all(np.isfinite(y)):
            raise ValueError("y_history must be finite and aligned with X_history.")
        self._set_features(X.shape[1], self.feature_names if self.feature_names else None)
        Xs = self.predictive_scaler.fit_transform(X)
        self.regressors = []
        for seed in self.random_seeds:
            model = RandomForestRegressor(
                n_estimators=self.n_trees, random_state=seed, n_jobs=-1,
            )
            model.fit(Xs, y)
            self.regressors.append(model)
        self._card("predictive_regression", len(X), training_source,
                   {"n_trees": self.n_trees, "interval": "ensemble_empirical_p05_p95"})
        return self

    def predict_with_uncertainty(self, x: np.ndarray) -> AIPrediction:
        if not self.regressors:
            raise RuntimeError("Call fit_predictive_ensemble() first.")
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != len(self.feature_names) or not np.all(np.isfinite(x)):
            raise ValueError("x is incompatible with the AI feature contract.")
        predictions = np.asarray(
            [m.predict(self.predictive_scaler.transform(x))[0] for m in self.regressors]
        )
        mean = float(np.mean(predictions))
        lower, upper = np.percentile(predictions, [5, 95])
        spread = float(np.std(predictions))
        confidence = float(np.clip(1.0 / (1.0 + spread), 0.0, 1.0))
        result = AIPrediction(mean, float(lower), float(upper), confidence, "ensemble_empirical_p05_p95")
        result.validate()
        return result

    def build_inference_record(
        self, *, acquisition_id: str, source: str, x: np.ndarray,
        assessment: AIAnomalyAssessment,
    ) -> AIInferenceRecord:
        """Create an auditable inference record for the anomaly task."""
        card = self.model_cards.get("unsupervised_anomaly")
        if card is None:
            raise RuntimeError("No unsupervised model card is available.")
        record = AIInferenceRecord(
            acquisition_id=acquisition_id,
            model_version=card.model_version,
            task=card.task,
            input_hash=_hash_array(x),
            source=source,
            output=(
                ("anomaly_score", assessment.anomaly_score),
                ("anomaly_vote_rate", assessment.anomaly_vote_rate),
                ("ensemble_agreement", assessment.ensemble_agreement),
            ),
            confidence=assessment.confidence,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        record.validate()
        return record
