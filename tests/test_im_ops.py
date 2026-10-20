"""Behavioural tests for the pixel arithmetic in `rcpyci.im_ops`."""

import numpy as np
import pytest

from rcpyci.im_ops import (
    apply_constant_scaling,
    apply_independent_scaling,
    apply_mask,
    apply_matched_scaling,
    combine,
    find_clusters,
    get_image_size,
)


def test_get_image_size_accepts_square_images():
    assert get_image_size(np.zeros((64, 64))) == 64


def test_get_image_size_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        get_image_size(np.zeros((32, 64)))


def test_get_image_size_rejects_non_2d():
    with pytest.raises(ValueError, match="2D"):
        get_image_size(np.zeros((8, 8, 3)))


def test_constant_scaling_uses_the_given_constant():
    ci = np.array([[-2.0, 0.0], [2.0, 1.0]])
    np.testing.assert_allclose(apply_constant_scaling(ci, 2.0), [[0.0, 0.5], [1.0, 0.75]])


def test_independent_scaling_spans_symmetric_extremes_and_ignores_nan():
    ci = np.array([[-4.0, np.nan], [4.0, 0.5]])
    scaled = apply_independent_scaling(ci)
    assert np.nanmin(scaled) == 0.0
    assert np.nanmax(scaled) == 1.0
    assert np.isnan(scaled[0, 1])


def test_matched_scaling_maps_ci_range_onto_base_range():
    base = np.array([[10.0, 12.0], [16.0, 20.0]])
    ci = np.array([[-1.0, 0.0], [1.0, 3.0]])
    scaled = apply_matched_scaling(ci, base)
    assert scaled.min() == pytest.approx(10.0)
    assert scaled.max() == pytest.approx(20.0)


def test_combine_is_the_mean_of_two_images():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_allclose(combine(a, b), [[2.0, 3.0], [4.0, 5.0]])


def test_apply_mask_masks_where_the_mask_is_zero():
    ci = np.ones((4, 4))
    mask = np.ones((4, 4))
    mask[:2] = 0
    masked = apply_mask(ci, mask)
    assert masked.mask[:, 0].tolist() == [True, True, False, False]
    assert not np.ma.is_masked(masked[3, 3])


def test_apply_mask_rejects_a_non_image_mask():
    with pytest.raises(ValueError, match="neither a path"):
        apply_mask(np.ones((4, 4)), np.ones(4))


def test_find_clusters_drops_clusters_below_min_size():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True  # 4 pixels
    mask[6, 6] = True  # 1 pixel
    kept = find_clusters(mask, min_size=2)
    assert kept[1:3, 1:3].all()
    assert not kept[6, 6]
    assert kept.sum() == 4
