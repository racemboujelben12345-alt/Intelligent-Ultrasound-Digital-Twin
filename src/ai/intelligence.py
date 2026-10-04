"""AI intelligence layer for the Intelligent Ultrasound Digital Twin.

Design goals
------------
- combine complementary anomaly learners instead of relying on one detector;
- expose model agreement as evidence confidence;
- support supervised learning when validated labels exist;
- support unsupervised operation when labels do not exist;
- provide bounded, auditable outputs;
- never interpret an AI anomaly as proof of physical hardware failure.

The AI layer consumes the canonical Digital Signature vector. It does not
replace provenance, statistical baselines, V&V or the physical validation layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AIAnomalyAssessment:
    """Structured AI anomaly assessment."""

    anomaly_score: float
    anomaly_probability: float
    confidence: float
    ensemble_agreement: float
    model_scores: tuple[float, ...]
    state: str

    def validate(self) -> None:
        values = (
            self.anomaly_score,
            self.anomaly_probability,
            self.confidence,
            self.ensemble_agreement,
        )
        if not all(np.isfinite(v) for v in values):
            raise ValueError("AI assessment contains non-finite values.")
        if not 0.0 <= self.anomaly_score <= 1.0:
            raise ValueError("anomaly_score must be in [0, 1].")
        if not 0.0 <= self.anomaly_probability <= 1.0:
            raise ValueError("anomaly_probability must be in [0, 1].")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        if not 0.0 <= self.ensemble_agreement <= 1.0:
            raise ValueError("ensemble_agreement must be in [0, 1].")
        if self.state not in {"NOMINAL", "WATCH", "ANOMALY"}:
            raise ValueError("Invalid AI state.")


@dataclass(frozen=True)
class AIPrediction:
    """Point prediction plus empirical ensemble uncertainty."""

    prediction: float
    lower: float
    upper: float
    confidence: float

    def validate(self) -> None:
        if not all(
            np.isfinite(v)
            for v in (self.prediction, self.lower, self.upper, self.confidence)
        ):
            raise ValueError("Prediction contains non-finite values.")
        if self.lower > self.prediction or self.prediction > self.upper:
            raise ValueError("Invalid prediction interval.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")


class UltrasoundAIEngine:
    """Expert-level ensemble AI layer for Digital Twin intelligence.

    Unsupervised mode:
        Standardized Digital Signature
          -> Isolation Forest ensemble
          -> consensus / uncertainty

    Supervised mode:
        Standardized Digital Signature
          -> Random Forest classifier
          -> class probability + feature importance

    Prediction mode:
        Feature/state history
          -> Random Forest regression ensemble
          -> empirical prediction interval

    All scores are model evidence, not physical-failure probabilities.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 7,
        random_seeds: tuple[int, ...] = (11, 23, 37, 41, 53, 67, 79),
        contamination: float = "auto",
    ) -> None:
        if n_estimators < 3:
            raise ValueError("n_estimators must be >= 3.")
        if not random_seeds:
            raise ValueError("At least one random seed is required.")

        self.n_estimators = int(n_estimators)
        self.random_seeds = tuple(random_seeds)
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.models: list[IsolationForest] = []
        self.supervised_model: RandomForestClassifier | None = None
        self.regressors: list[RandomForestRegressor] = []
        self.feature_names: tuple[str, ...] = ()
        self.fitted = False

    @staticmethod
    def _matrix(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must contain at least two observations.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf.")
        return X

    def fit_reference(
        self,
        X_reference: np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> "UltrasoundAIEngine":
        """Fit the unsupervised anomaly ensemble on normal/reference data."""
        X = self._matrix(X_reference)
        Xs = self.scaler.fit_transform(X)

        self.models = []
        for seed in self.random_seeds:
            model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(Xs)
            self.models.append(model)

        self.feature_names = tuple(
            feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        )
        if len(self.feature_names) != X.shape[1]:
            raise ValueError("feature_names length does not match X.")
        self.fitted = True
        return self

    def fit_supervised(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "UltrasoundAIEngine":
        """Train a validated-label classifier when labels are available.

        Labels should represent engineering states or validated simulation
        classes, not unsupported claims of physical failure.
        """
        X = self._matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1D array aligned with X.")
        if len(np.unique(y)) < 2:
            raise ValueError("Supervised AI requires at least two classes.")

        Xs = self.scaler.fit_transform(X)
        self.supervised_model = RandomForestClassifier(
            n_estimators=max(200, self.n_estimators * 30),
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.supervised_model.fit(Xs, y)
        self.feature_names = tuple(
            self.feature_names
            or [f"feature_{i}" for i in range(X.shape[1])]
        )
        return self

    def assess(self, x: np.ndarray) -> AIAnomalyAssessment:
        """Return consensus anomaly evidence from the ensemble."""
        if not self.fitted or not self.models:
            raise RuntimeError("Call fit_reference() before assess().")

        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN or Inf.")

        xs = self.scaler.transform(x)
        raw = np.asarray(
            [-float(model.score_samples(xs)[0]) for model in self.models]
        )

        # Robustly map the ensemble score to [0, 1].
        q05, q95 = np.percentile(raw, [5, 95])
        scale = max(float(q95 - q05), 1e-9)
        score = float(np.clip((np.mean(raw) - q05) / scale, 0.0, 1.0))

        votes = np.asarray(
            [model.predict(xs)[0] == -1 for model in self.models],
            dtype=bool,
        )
        probability = float(np.mean(votes))
        agreement = float(max(probability, 1.0 - probability))
        confidence = float(np.clip(0.5 * agreement + 0.5 * (1.0 - np.std(raw) / (np.mean(np.abs(raw)) + 1e-9)), 0.0, 1.0))

        state = (
            "ANOMALY" if probability >= 0.67 or score >= 0.80
            else "WATCH" if probability >= 0.34 or score >= 0.50
            else "NOMINAL"
        )

        result = AIAnomalyAssessment(
            anomaly_score=score,
            anomaly_probability=probability,
            confidence=confidence,
            ensemble_agreement=agreement,
            model_scores=tuple(float(v) for v in raw),
            state=state,
        )
        result.validate()
        return result

    def predict_class(self, x: np.ndarray) -> dict:
        """Predict an engineering class with probability and importance."""
        if self.supervised_model is None:
            raise RuntimeError("Call fit_supervised() first.")

        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        probabilities = self.supervised_model.predict_proba(
            self.scaler.transform(x)
        )[0]
        index = int(np.argmax(probabilities))

        return {
            "class": self.supervised_model.classes_[index].item(),
            "probability": float(probabilities[index]),
            "probabilities": {
                str(label): float(prob)
                for label, prob in zip(
                    self.supervised_model.classes_, probabilities
                )
            },
            "feature_importance": {
                name: float(value)
                for name, value in zip(
                    self.feature_names,
                    self.supervised_model.feature_importances_,
                )
            },
        }

    def fit_predictive_ensemble(
        self,
        X_history: np.ndarray,
        y_history: np.ndarray,
    ) -> "UltrasoundAIEngine":
        """Fit an ensemble regressor for a future engineering indicator."""
        X = self._matrix(X_history)
        y = np.asarray(y_history, dtype=np.float64)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y_history must align with X_history.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y_history contains NaN or Inf.")

        Xs = self.scaler.fit_transform(X)
        self.regressors = []
        for seed in self.random_seeds:
            model = RandomForestRegressor(
                n_estimators=max(150, self.n_estimators * 25),
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(Xs, y)
            self.regressors.append(model)
        return self

    def predict_with_uncertainty(self, x: np.ndarray) -> AIPrediction:
        """Predict an engineering indicator with ensemble interval."""
        if not self.regressors:
            raise RuntimeError("Call fit_predictive_ensemble() first.")

        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN or Inf.")

        xs = self.scaler.transform(x)
        predictions = np.asarray(
            [model.predict(xs)[0] for model in self.regressors],
            dtype=np.float64,
        )
        mean = float(np.mean(predictions))
        lower, upper = np.percentile(predictions, [5, 95])
        spread = float(np.std(predictions))
        confidence = float(np.clip(1.0 / (1.0 + spread), 0.0, 1.0))

        result = AIPrediction(
            prediction=mean,
            lower=float(lower),
            upper=float(upper),
            confidence=confidence,
        )
        result.validate()
        return result
