import numpy as np

from src.ai.intelligence import UltrasoundAIEngine


def _reference(seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(80, 13))


def test_unsupervised_ensemble_is_bounded_and_deterministic():
    X = _reference()
    engine = UltrasoundAIEngine(random_seeds=(11, 23, 37))
    engine.fit_reference(X)

    x = X[0]
    a = engine.assess(x)
    b = engine.assess(x)

    assert 0.0 <= a.anomaly_score <= 1.0
    assert 0.0 <= a.anomaly_probability <= 1.0
    assert 0.0 <= a.confidence <= 1.0
    assert 0.0 <= a.ensemble_agreement <= 1.0
    assert a == b


def test_supervised_mode_returns_class_probability_and_importance():
    X = _reference()
    y = np.array(["nominal"] * 40 + ["degraded"] * 40)
    X[40:] += 2.0

    engine = UltrasoundAIEngine(random_seeds=(11, 23, 37))
    engine.fit_supervised(X, y)

    result = engine.predict_class(X[-1])

    assert result["class"] in {"nominal", "degraded"}
    assert 0.0 <= result["probability"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-9
    assert len(result["feature_importance"]) == X.shape[1]


def test_predictive_ensemble_exposes_uncertainty():
    X = _reference()
    y = X[:, 0] * 2.0 - X[:, 1] + 0.1 * X[:, 2]

    engine = UltrasoundAIEngine(random_seeds=(11, 23, 37))
    engine.fit_predictive_ensemble(X, y)
    result = engine.predict_with_uncertainty(X[0])

    assert result.lower <= result.prediction <= result.upper
    assert 0.0 <= result.confidence <= 1.0
