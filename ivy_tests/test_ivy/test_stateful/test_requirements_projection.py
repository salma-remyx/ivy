"""Tests for the RequirementsProjection layer (adapted from PiShield).

These tests reach the layer through the existing ``ivy.stateful`` public API
(``from ivy.stateful.layers import RequirementsProjection``) and assert the
core guarantee: the projected output satisfies every requirement for any input.
"""

import numpy as np
import pytest

import ivy
from ivy.stateful.layers import RequirementsProjection


@pytest.fixture
def numpy_backend():
    ivy.set_backend("numpy")
    yield
    ivy.previous_backend()


def _violation(layer, weight, x):
    # max |A @ y - midline| measured against equality targets / clamp bounds.
    y = layer(x)
    scores = ivy.matmul(weight, ivy.swapaxes(y, -1, -2))
    return y, scores


def test_equality_requirements_satisfied(numpy_backend):
    # Two equality requirements: y0 + y1 + y2 == 5 and y0 - y1 == 1.
    weight = ivy.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]], dtype="float64")
    target = ivy.array([5.0, 1.0], dtype="float64")
    layer = RequirementsProjection(weight, lower=target, upper=target, dtype="float64")

    x = ivy.array(
        [[10.0, -2.0, 0.5], [0.1, 0.2, 0.3], [4.0, 4.0, 4.0]], dtype="float64"
    )
    _, scores = _violation(layer, weight, x)
    # Every column (one per sample) must match the target for both requirements.
    max_err = float(ivy.max(ivy.abs(scores - target[:, None])))
    assert max_err < 1e-6


def test_equality_projection_is_idempotent(numpy_backend):
    # Orthogonal projection is idempotent: re-projecting an already feasible
    # point leaves it unchanged (P(P(x)) == P(x)).
    weight = ivy.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]], dtype="float64")
    target = ivy.array([5.0, 1.0], dtype="float64")
    layer = RequirementsProjection(weight, lower=target, upper=target, dtype="float64")
    x = ivy.array([[9.0, -3.0, 2.0], [0.0, 10.0, -5.0]], dtype="float64")
    projected = layer(x)
    reprojected = layer(projected)
    assert float(ivy.max(ivy.abs(reprojected - projected))) < 1e-9


def test_equality_projection_is_nearest_point(numpy_backend):
    # The projection is the closest feasible point: ||x - P(x)|| <= ||x - z||
    # for any other feasible point z.
    weight = ivy.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]], dtype="float64")
    target = ivy.array([5.0, 1.0], dtype="float64")
    layer = RequirementsProjection(weight, lower=target, upper=target, dtype="float64")
    x = ivy.array([[9.0, -3.0, 2.0]], dtype="float64")
    projected = layer(x)
    dist_proj = float(ivy.sqrt(ivy.sum((x - projected) ** 2)))
    # Any other feasible point z (here a different feasible solution) is farther.
    z = ivy.array([[3.0, 2.0, 0.0]], dtype="float64")  # satisfies sum==5, y0-y1==1
    dist_z = float(ivy.sqrt(ivy.sum((x - z) ** 2)))
    assert dist_proj < dist_z


def test_bound_requirements_satisfied(numpy_backend):
    # 0 <= y0 + y1 + y2 <= 10, and 0 <= y0 <= 2.
    weight = ivy.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype="float64")
    lower = ivy.array([0.0, 0.0], dtype="float64")
    upper = ivy.array([10.0, 2.0], dtype="float64")
    layer = RequirementsProjection(
        weight, lower=lower, upper=upper, num_iterations=500, dtype="float64"
    )

    x = ivy.array(
        [[-5.0, -5.0, -5.0], [50.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype="float64"
    )
    y, scores = _violation(layer, weight, x)
    scores_np = ivy.to_numpy(scores)
    assert np.all(scores_np >= -1e-6)
    assert np.all(scores_np <= ivy.to_numpy(upper)[:, None] + 1e-6)
    assert y.shape == x.shape


def test_one_sided_bound_clamps(numpy_backend):
    # y0 >= 0 with a single requirement row.
    weight = ivy.array([[1.0, 0.0, 0.0]], dtype="float64")
    layer = RequirementsProjection(
        weight, lower=ivy.array([0.0]), upper=None, num_iterations=200, dtype="float64"
    )
    x = ivy.array([[-3.0, 5.0, 2.0], [0.4, 1.0, 1.0]], dtype="float64")
    y = layer(x)
    out = ivy.to_numpy(y)
    assert np.all(out[..., 0] >= -1e-12)
    # Non-clamped entries are untouched for an axis-aligned requirement.
    assert float(np.max(np.abs(out[1] - ivy.to_numpy(x[1])))) == 0.0


def test_no_requirements_is_identity(numpy_backend):
    # Finite bounds default to (-inf, +inf): no constraint -> identity.
    weight = ivy.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0]], dtype="float64")
    layer = RequirementsProjection(weight, dtype="float64")
    x = ivy.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]], dtype="float64")
    assert float(ivy.max(ivy.abs(layer(x) - x))) == 0.0


def test_invalid_construction_raises(numpy_backend):
    with pytest.raises(ivy.utils.exceptions.IvyValueError):
        RequirementsProjection(ivy.array([1.0, 2.0, 3.0]))  # 1D weight
    weight = ivy.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]], dtype="float64")
    with pytest.raises(ivy.utils.exceptions.IvyValueError):
        RequirementsProjection(
            weight, lower=ivy.array([1.0, 2.0, 3.0]), dtype="float64"
        )  # wrong bound shape
    with pytest.raises(ivy.utils.exceptions.IvyValueError):
        RequirementsProjection(
            weight,
            lower=ivy.array([0.0, 0.0]),
            upper=ivy.array([0.0, 0.0]),
            num_iterations=0,
        )  # non-positive iterations


def test_layer_repr(numpy_backend):
    weight = ivy.array([[1.0, 1.0, 1.0]], dtype="float64")
    layer = RequirementsProjection(
        weight, lower=ivy.array([1.0]), upper=ivy.array([1.0]), num_iterations=7
    )
    rep = repr(layer)
    assert "requirements=1" in rep
    assert "num_iterations=7" in rep
