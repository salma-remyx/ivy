"""Tests for the weight conditioning (diagonal equilibration) layer.

These tests exercise the public ``WeightConditioning`` module and the
``weight_conditioning`` functional helper through the existing
``ivy.stateful.norms`` module (the call site), asserting the core result of
the paper: row equilibration narrows the spread of a matrix's singular
values, i.e. reduces its condition number.
"""

# global
import numpy as np
import pytest

# local
import ivy
from ivy.stateful.norms import WeightConditioning, weight_conditioning


@pytest.fixture(autouse=True)
def _numpy_backend():
    """Pin a backend for the duration of each test and restore the prior one."""
    ivy.set_backend("numpy")
    yield
    ivy.unset_backend()


def _condition_number(matrix):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return singular_values.max() / singular_values.min()


def test_row_equilibration_produces_unit_norm_rows():
    # An ill-conditioned weight matrix: rows span very different magnitudes.
    weights = ivy.asarray(
        np.array(
            [
                [10.0, -8.0, 4.0, 2.0],
                [0.1, 0.2, -0.1, 0.05],
                [3.0, 3.0, 3.0, 3.0],
            ]
        )
    )

    conditioned = weight_conditioning(weights, mode="row")

    # Shape is preserved by the tensor -> tensor forward contract.
    assert tuple(conditioned.shape) == tuple(weights.shape)

    # Every row is scaled to unit Euclidean norm -- the defining property.
    row_norms = ivy.to_numpy(ivy.vector_norm(conditioned, axis=-1))
    assert np.allclose(row_norms, 1.0, atol=1e-6)


def test_module_forward_matches_helper_and_exports_publicly():
    weights = ivy.asarray(np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0], [-2.0, 2.0]]))

    module = WeightConditioning(mode="row")
    via_module = module(weights)
    via_helper = weight_conditioning(weights, mode="row")

    assert np.allclose(ivy.to_numpy(via_module), ivy.to_numpy(via_helper))

    # The class is reachable through the public (non-new) stateful namespace.
    from ivy.stateful import WeightConditioning as PublicWeightConditioning

    assert PublicWeightConditioning is WeightConditioning


def test_zero_row_is_safe_from_division_by_zero():
    weights = ivy.asarray(np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
    conditioned = ivy.to_numpy(weight_conditioning(weights, mode="row"))

    assert not np.any(np.isnan(conditioned))
    # A zero row stays zero rather than exploding to inf.
    assert np.allclose(conditioned[0], 0.0)


def test_row_equilibration_reduces_condition_number():
    # Row magnitudes differ by ~3 orders of magnitude -> badly conditioned.
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((5, 5))
    raw *= np.array([[1000.0], [1.0], [10.0], [0.01], [100.0]])
    weights_np = raw

    conditioned_np = ivy.to_numpy(
        weight_conditioning(ivy.asarray(weights_np), mode="row")
    )

    before = _condition_number(weights_np)
    after = _condition_number(conditioned_np)

    # The paper's core claim: equilibration narrows the singular-value spread.
    assert after < before
