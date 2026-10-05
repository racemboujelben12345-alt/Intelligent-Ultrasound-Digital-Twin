from src.physics.causal_evidence import rank_causal_evidence, score_causal_evidence


def test_focus_directional_evidence():
    result = score_causal_evidence(
        "focus_shift_or_defocus",
        {
            "sharpness_laplacian": -0.4,
            "edge_density": -0.2,
            "gradient_mean": -0.1,
        },
    )

    assert result.agreement_score == 1.0
    assert len(result.supported_features) == 3
    assert not result.contradicted_features


def test_contradiction_is_not_hidden():
    result = score_causal_evidence(
        "focus_shift_or_defocus",
        {
            "sharpness_laplacian": 0.4,
            "edge_density": -0.2,
            "gradient_mean": -0.1,
        },
    )

    assert result.agreement_score < 1.0
    assert "sharpness_laplacian" in result.contradicted_features


def test_ranking_is_deterministic():
    delta = {
        "sharpness_laplacian": -0.3,
        "edge_density": -0.2,
        "gradient_mean": -0.1,
        "std_intensity": 0.2,
        "coefficient_variation": 0.1,
    }

    first = rank_causal_evidence(delta)
    second = rank_causal_evidence(delta)

    assert first == second
    assert first[0].agreement_score >= first[-1].agreement_score
