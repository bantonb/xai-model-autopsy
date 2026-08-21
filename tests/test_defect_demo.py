"""Assert that the synthetic demo reproduces the coupling defect and that
adding [1,1] / [0,0] labels removes it. Seed is fixed inside build_arm."""
import pytest

from defect_demo.run import SEED, build_arm


@pytest.fixture(scope="module")
def broken():
    _, d = build_arm("broken", seed=SEED)
    return d


@pytest.fixture(scope="module")
def corrected():
    _, d = build_arm("corrected", seed=SEED)
    return d


def test_broken_arm_couples_outputs(broken):
    # Two sigmoid outputs almost perfectly anti-correlated across the test grid.
    assert abs(broken.pearson_r_probs) > 0.95, (
        f"broken arm should have |r(p0,p1)| > 0.95, got r = {broken.pearson_r_probs:+.4f}"
    )
    # And p0 + p1 sits essentially on 1 everywhere.
    assert broken.mean_abs_p0_plus_p1_minus_1 < 0.05, (
        f"broken arm should have mean|p0+p1-1| < 0.05, got "
        f"{broken.mean_abs_p0_plus_p1_minus_1:.4f}"
    )


def test_corrected_arm_decouples_outputs(corrected):
    # Adding [1,1] and [0,0] labels breaks both symptoms.
    assert abs(corrected.pearson_r_probs) <= 0.95, (
        f"corrected arm should have |r(p0,p1)| <= 0.95, got "
        f"r = {corrected.pearson_r_probs:+.4f}"
    )
    assert corrected.mean_abs_p0_plus_p1_minus_1 >= 0.05, (
        f"corrected arm should have mean|p0+p1-1| >= 0.05, got "
        f"{corrected.mean_abs_p0_plus_p1_minus_1:.4f}"
    )


def test_broken_attributions_mirrored(broken):
    # FeatureAblation attributions to class 0 vs class 1 are near-mirror images.
    assert broken.pearson_r_attr < -0.9, (
        f"broken arm attributions should be strongly anti-correlated, "
        f"got r(attr0, attr1) = {broken.pearson_r_attr:+.4f}"
    )


def test_corrected_attributions_not_mirrored(corrected):
    # In the corrected arm attributions are no longer forced to mirror.
    assert corrected.pearson_r_attr > -0.9, (
        f"corrected arm attributions should not be strongly anti-correlated, "
        f"got r(attr0, attr1) = {corrected.pearson_r_attr:+.4f}"
    )
