"""Numeric parity between `rcpyci.core.compute_ci` and R's `rcicr::generateCI`.

`ref/` holds the full R-vs-python comparison, but it needs R, rpy2 and a 4 GB container,
so it cannot gate anything. This test carries the result of that comparison as a golden
fixture instead: R's classification image for a fixed seeded input, checked against the
current implementation on every push in well under a second.

Regenerate it (only when the reference implementation or the R package changes):

    docker run --rm -v "$PWD/ref":/tests -v "$PWD":/repo -w /tests \\
      hvalev/rcpyci:latest /bin/bash /tests/run_golden_fixtures.sh

A real algorithmic change moves the deviation by orders of magnitude (O(1e-2)), the
floating point noise between two correct implementations sits at O(1e-16), so 1e-13 is
far above the noise and far below any meaningful drift.
"""

import pathlib

import numpy as np
import pytest

from rcpyci.core import compute_ci
from rcpyci.im_ops import apply_independent_scaling
from rcpyci.pipelines import compute_zscore_ci

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "rcicr_seed1.npz"
ATOL = 1e-13
Z_ATOL = 1e-3  # R and scipy blur at slightly different last digits
Z_NAN_TOLERANCE = 64  # pixels that fall on the threshold boundary differently


@pytest.fixture(scope="module")
def golden():
    # Fails rather than skips: the fixture is committed, so a missing file means a
    # broken checkout, and a silently skipping parity test would read as a pass.
    if not FIXTURE.exists():
        pytest.fail(f"{FIXTURE.name} is missing - regenerate it with ref/run_golden_fixtures.sh")
    return np.load(FIXTURE, allow_pickle=True)


@pytest.fixture(scope="module")
def ci(golden):
    """Current implementation's CI for the same input R used.

    compute_ci only reads the base image's size; patches and patch indices are regenerated
    internally from n_scales/noise_type (rcpyci counts patches from 0, so nothing here may
    pass R's 1-based indices through).
    """
    img_size = int(golden["meta"].item()["img_size"])
    return compute_ci(
        base_image=np.zeros((img_size, img_size)),
        responses=golden["responses"],
        stimuli_params=golden["stimuli_params"],
        n_scales=5,
        noise_type="sinusoid",
    )


def summary(array):
    array = np.asarray(array, dtype=float)
    return np.array([np.nanmean(array), np.nanstd(array), float(np.isnan(array).sum())])


def test_fixture_contract(golden):
    """Guard against a corrupted or mis-generated fixture."""
    assert golden["stimuli_params"].shape == (5, 4092)
    assert golden["responses"].shape == (5, 1)
    assert golden["r_ci"].ndim == 2 and golden["r_ci"].shape[0] == golden["r_ci"].shape[1]
    for key in ("r_scaled_stats", "r_zmap_stats"):
        assert golden[key].shape == (3,)
        assert np.isfinite(golden[key]).all()
    meta = golden["meta"].item()
    assert meta["source"] == "R rcicr generateCI()"
    assert meta["noise_type"] == "sinusoid"


def test_compute_ci_matches_rcicr(golden, ci):
    np.testing.assert_allclose(ci, golden["r_ci"], rtol=0.0, atol=ATOL)


def test_scaled_image_matches_rcicr(golden, ci):
    """Scaling a CI to a viewable range has to keep R's convention."""
    observed = summary(apply_independent_scaling(ci))
    np.testing.assert_allclose(observed[:2], golden["r_scaled_stats"][:2], rtol=0.0, atol=ATOL)
    assert observed[2] == golden["r_scaled_stats"][2]


def test_quick_zmap_matches_rcicr(golden, ci):
    """The quick z-map is blur, then z-score, then blank |z| < threshold."""
    zmap = compute_zscore_ci(ci, sigma=3)["zscore_image"]
    zmap = np.where((zmap > -1) & (zmap < 1), np.nan, zmap)
    observed = summary(zmap)
    expected = golden["r_zmap_stats"]
    np.testing.assert_allclose(observed[:2], expected[:2], rtol=0.0, atol=Z_ATOL)
    assert abs(observed[2] - expected[2]) <= Z_NAN_TOLERANCE
