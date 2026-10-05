from src.digital_twin.evidence_fusion import fuse_evidence


def test_fusion_is_bounded():
    result = fuse_evidence(
        statistical=0.8,
        ai=0.9,
        physics=0.7,
        causal=0.8,
        counterfactual=0.6,
    )

    assert 0.0 <= result.fusion_score <= 1.0
    assert 0.0 <= result.disagreement <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_agreement_increases_confidence():
    same = fuse_evidence(
        statistical=0.8, ai=0.8, physics=0.8, causal=0.8, counterfactual=0.8
    )
    mixed = fuse_evidence(
        statistical=1.0, ai=0.0, physics=1.0, causal=0.0, counterfactual=1.0
    )

    assert same.disagreement < mixed.disagreement
    assert same.confidence > mixed.confidence
