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

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "rcicr_seed1.npz"
ATOL = 1e-13


@pytest.fixture(scope="module")
def golden():
    # Fails rather than skips: the fixture is committed, so a missing file means a
    # broken checkout, and a silently skipping parity test would read as a pass.
    if not FIXTURE.exists():
        pytest.fail(f"{FIXTURE.name} is missing - regenerate it with ref/run_golden_fixtures.sh")
    return np.load(FIXTURE, allow_pickle=True)


def test_fixture_contract(golden):
    """Guard against a corrupted or mis-generated fixture."""
    assert golden["stimuli_params"].shape == (5, 4092)
    assert golden["responses"].shape == (5, 1)
    assert golden["r_ci"].ndim == 2 and golden["r_ci"].shape[0] == golden["r_ci"].shape[1]
    meta = golden["meta"].item()
    assert meta["source"] == "R rcicr generateCI()"
    assert meta["noise_type"] == "sinusoid"


def test_compute_ci_matches_rcicr(golden):
    params = golden["stimuli_params"]
    responses = golden["responses"]
    img_size = int(golden["meta"].item()["img_size"])

    # compute_ci only reads the base image's size; patches and patch indices are
    # regenerated internally from n_scales/noise_type (rcpyci counts patches from 0,
    # so nothing here may pass R's 1-based indices through).
    ci = compute_ci(
        base_image=np.zeros((img_size, img_size)),
        responses=responses,
        stimuli_params=params,
        n_scales=5,
        noise_type="sinusoid",
    )

    np.testing.assert_allclose(ci, golden["r_ci"], rtol=0.0, atol=ATOL)
