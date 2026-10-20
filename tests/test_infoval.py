"""Tests for the informational-value statistics in `rcpyci.infoval`."""

import numpy as np
import pytest

from rcpyci.infoval import compute_info_val_2ifc, generate_reference_distribution_2ifc

SMALL = dict(n_trials=5, n_scales=5, iter=6)


def test_compute_info_val_2ifc_matches_its_definition():
    reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = np.array([[3.0, 4.0]])  # Frobenius norm 5

    info_val, cinorm, ref_median, ref_mad, ref_iter = compute_info_val_2ifc(target, reference)

    assert cinorm == pytest.approx(np.linalg.norm(target, "f"))
    assert ref_median == pytest.approx(3.0)
    assert ref_mad == pytest.approx(1.0)  # median(|reference - 3|)
    assert ref_iter == 5
    assert info_val == pytest.approx((5.0 - 3.0) / 1.0)


def test_reference_distribution_is_reproducible_when_numpy_is_seeded():
    """`seed` only pins the stimulus parameters.

    The simulated responses come from the global numpy RNG, so a caller who wants a
    reproducible reference distribution has to seed numpy themselves. This test pins the
    behaviour rather than changing it, because published results were produced this way.
    """
    base = np.zeros((512, 512))

    np.random.seed(7)
    first = generate_reference_distribution_2ifc(base, **SMALL)
    np.random.seed(7)
    second = generate_reference_distribution_2ifc(base, **SMALL)

    assert first.shape == (SMALL["iter"],)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert np.all(first > 0)


def test_reference_distribution_warns_about_small_iteration_counts(capsys):
    generate_reference_distribution_2ifc(np.zeros((512, 512)), **SMALL)
    assert "iter >= 10000" in capsys.readouterr().out
