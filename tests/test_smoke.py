"""Smoke tests for the parts of the package that must work on every supported Python."""

import importlib

import numpy as np
import pytest

from rcpyci import im_ops, utils

MODULES = ["core", "im_ops", "infoval", "interface", "pipelines", "utils", "visualise"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"rcpyci.{name}")


def test_save_and_read_image_roundtrip(tmp_path):
    image = np.linspace(0, 1, 64 * 64, dtype=np.float64).reshape(64, 64)
    path = tmp_path / "stim.png"
    im_ops.save_image(image, str(path))
    read_back = im_ops.read_image(str(path))
    assert read_back.shape == (64, 64)
    assert read_back.min() >= 0 and read_back.max() <= 1


def test_create_test_data_shape():
    data = utils.create_test_data(n_participants=2, n_trials=5, seed=1)
    for column in ("participant_id", "condition", "stimulus_id", "responses"):
        assert column in data.columns
    assert len(data) == 10
    assert set(data["responses"].unique()) <= {-1, 1}
