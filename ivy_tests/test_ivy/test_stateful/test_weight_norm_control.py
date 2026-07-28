"""Integration tests for the AdamWN (weight norm control) optimizer.

These tests import :class:`AdamW` and :class:`AdamWN` from the existing
``ivy.stateful.optimizers`` module (not the new capability module) and exercise
the wiring added there: :class:`AdamWN` subclasses :class:`AdamW` and replaces
its weight-decay term with weight-norm-control decay toward a target L2 norm,
following "Weight Norm Control" (arxiv:2311.11446).
"""

# global
import numpy as np
import pytest

# local
import ivy
from ivy.stateful.optimizers import AdamW, AdamWN

_TARGET_KWARGS = dict(lr=0.5, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=1.0)


@pytest.fixture
def numpy_backend():
    ivy.set_backend("numpy")
    yield
    ivy.previous_backend()


def _run(opt, steps=10):
    """Step ``opt`` repeatedly with fresh zero gradients.

    Zero gradients isolate the weight-decay / norm-control term: the only force
    acting on the parameter is the decay toward (or away from) the target norm.
    """
    v = ivy.Container(a=ivy.array([3.0, 4.0]))  # L2 norm == 5
    for _ in range(steps):
        v = opt.step(v, ivy.Container(a=ivy.array([0.0, 0.0])))
    return v


def test_adamwn_target_zero_matches_adamw(numpy_backend):
    # target_norm=0 must reproduce AdamW exactly: weight decay is the special
    # case of weight-norm-control with target norm 0.
    out_w = _run(AdamW(**_TARGET_KWARGS))
    out_n = _run(AdamWN(target_norm=0.0, **_TARGET_KWARGS))
    np.testing.assert_allclose(ivy.to_numpy(out_n.a), ivy.to_numpy(out_w.a), atol=1e-6)


def test_adamwn_nonzero_target_decays_less_than_zero(numpy_backend):
    # Starting norm 5 with target 2 shrinks the parameter, but less than decay
    # toward 0 since the target is closer to the current norm.
    n_zero = float(ivy.vector_norm(_run(AdamWN(target_norm=0.0, **_TARGET_KWARGS)).a))
    n_two = float(ivy.vector_norm(_run(AdamWN(target_norm=2.0, **_TARGET_KWARGS)).a))
    assert n_two > n_zero  # pulled toward 2, not 0
    assert n_two < 5.0  # still shrunk from the initial norm of 5


def test_adamwn_target_above_norm_grows_parameter(numpy_backend):
    # A target norm above the current norm pushes the parameter away from zero.
    n_eight = float(ivy.vector_norm(_run(AdamWN(target_norm=8.0, **_TARGET_KWARGS)).a))
    assert n_eight > 5.0  # grew from the initial norm of 5 toward 8


def test_adamwn_at_target_leaves_parameter_unchanged(numpy_backend):
    # When the parameter already has the target norm the control term vanishes,
    # so (under zero gradient) it does not move.
    out = _run(AdamWN(target_norm=5.0, **_TARGET_KWARGS))
    np.testing.assert_allclose(ivy.to_numpy(out.a), np.array([3.0, 4.0]), atol=1e-6)
