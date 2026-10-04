import numpy as np

from src.ai.intelligence import UltrasoundAIEngine


def _reference(seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(80, 13))


def test_unsupervised_ensemble_is_bounded_and_deterministic():
    X = _reference()
    engine = UltrasoundAIEngine(n_models=3, random_seeds=(11, 23, 37))
    engine.fit_reference(X, training_source="test_reference")

    a = engine.assess(X[0])
    b = engine.assess(X[0])

    assert 0.0 <= a.anomaly_score <= 1.0
    assert 0.0 <= a.anomaly_vote_rate <= 1.0
    assert a.anomaly_probability == a.anomaly_vote_rate
    assert 0.0 <= a.confidence <= 1.0
    assert 0.0 <= a.ensemble_agreement <= 1.0
    assert a == b
    assert engine.model_cards["unsupervised_anomaly"].training_source == "test_reference"


def test_ai_feature_sensitivity_is_ranked_and_bounded():
    X = _reference()
    engine = UltrasoundAIEngine(n_models=3, random_seeds=(11, 23, 37))
    engine.fit_reference(X)

    x = X[0].copy()
    x[3] += 5.0
    contributions = engine.anomaly_feature_contributions(x)

    assert len(contributions) == X.shape[1]
    assert all(value >= 0.0 for _, value in contributions)
    assert all(np.isfinite(value) for _, value in contributions)


def test_supervised_and_predictive_paths_have_independent_scalers():
    X = _reference()
    y = np.array(["nominal"] * 40 + ["degraded"] * 40)
    X[40:] += 2.0

    engine = UltrasoundAIEngine(n_models=3, random_seeds=(11, 23, 37))
    engine.fit_reference(X)
    engine.fit_supervised(X, y)
    result = engine.predict_class(X[-1])

    assert result["class"] in {"nominal", "degraded"}
    assert 0.0 <= result["probability"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-9
    assert len(result["feature_importance"]) == X.shape[1]

    target = X[:, 0] * 2.0 - X[:, 1]
    engine.fit_predictive_ensemble(X, target)
    prediction = engine.predict_with_uncertainty(X[0])
    assert prediction.lower <= prediction.prediction <= prediction.upper


def test_inference_record_is_traceable():
    X = _reference()
    engine = UltrasoundAIEngine(n_models=3, random_seeds=(11, 23, 37))
    engine.fit_reference(X, training_source="synthetic_vv")
    assessment = engine.assess(X[0])

    record = engine.build_inference_record(
        acquisition_id="ACQ-001",
        source="simulated",
        x=X[0],
        assessment=assessment,
    )

    assert record.acquisition_id == "ACQ-001"
    assert record.task == "unsupervised_anomaly"
    assert len(record.input_hash) == 64
    assert 0.0 <= record.confidence <= 1.0


def test_anomaly_evidence_is_calibrated_against_reference_population():
    X = _reference()
    engine = UltrasoundAIEngine(n_models=3, random_seeds=(11, 23, 37))
    engine.fit_reference(X)

    nominal = engine.assess(X[0])
    degraded = engine.assess(X[0] + np.array([6.0] + [0.0] * 12))

    assert 0.0 <= nominal.anomaly_score <= 1.0
    assert 0.0 <= degraded.anomaly_score <= 1.0
    assert degraded.anomaly_score >= nominal.anomaly_score
    assert len(engine.reference_anomaly_scores) == len(X)
