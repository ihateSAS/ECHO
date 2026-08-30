from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROBE = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-08-13-encoder-level-probe/probe.py"
)


@pytest.fixture(scope="module")
def probe_module():
    spec = importlib.util.spec_from_file_location("encoder_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["encoder_probe"] = module
    spec.loader.exec_module(module)
    return module


def test_the_shuffle_actually_reorders_the_labels(probe_module) -> None:
    targets = np.arange(20, dtype=float)
    groups = np.repeat(np.arange(4), 5)
    shuffled = probe_module.shuffle_within_groups(targets, groups, seed=0)
    assert not np.array_equal(shuffled, targets)


def test_the_shuffle_stays_inside_each_group(probe_module) -> None:
    targets = np.arange(20, dtype=float)
    groups = np.repeat(np.arange(4), 5)
    shuffled = probe_module.shuffle_within_groups(targets, groups, seed=0)
    for group in np.unique(groups):
        mask = groups == group
        assert sorted(shuffled[mask]) == sorted(targets[mask])


def test_the_shuffle_does_not_modify_its_input(probe_module) -> None:
    targets = np.arange(20, dtype=float)
    original = targets.copy()
    probe_module.shuffle_within_groups(targets, np.repeat(np.arange(4), 5), seed=0)
    assert np.array_equal(targets, original)


def test_the_shuffle_is_reproducible(probe_module) -> None:
    targets = np.arange(20, dtype=float)
    groups = np.repeat(np.arange(4), 5)
    first = probe_module.shuffle_within_groups(targets, groups, seed=7)
    second = probe_module.shuffle_within_groups(targets, groups, seed=7)
    assert np.array_equal(first, second)


def test_a_probe_on_shuffled_labels_fails_on_separable_data(probe_module) -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 11)
    targets = np.tile(np.linspace(-25.0, 25.0, 11), 8)
    features = np.column_stack(
        [targets, rng.standard_normal((targets.size, 12))]
    )

    real = probe_module.leave_one_out_probe(features, targets, groups)
    null = probe_module.leave_one_out_probe(
        features, probe_module.shuffle_within_groups(targets, groups, 1), groups
    )

    assert real["r2"] > 0.95
    assert null["r2"] < 0.5
    assert null["mae_deg"] > real["mae_deg"] * 3
