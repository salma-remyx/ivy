"""Integration tests for the Weight-Conditioning optimizer.

These exercise the wiring in ``ivy.stateful.optimizers`` -- the call site
that invokes the row-equilibration operator from
``ivy.stateful.weight_conditioning`` inside ``Optimizer._step`` -- rather
than the operator in isolation.
"""

# global
import pytest

# local
import ivy
from ivy.stateful.optimizers import WeightConditioning

_BACKENDS = ["numpy", "torch", "tensorflow", "jax", "paddle", "mxnet"]


@pytest.fixture
def any_backend():
    """Activate the first available ivy backend, or skip if none is installed."""
    chosen = None
    for name in _BACKENDS:
        try:
            ivy.set_backend(name)
            chosen = name
            break
        except Exception:
            continue
    if chosen is None:
        pytest.skip("no ivy backend available")
    yield chosen
    ivy.previous_backend()


def test_step_equilibrates_weight_rows_and_leaves_bias_untouched(any_backend):
    """The optimizer's step row-equilibrates 2-D weights and passes 1-D
    biases through unchanged (proves the operator is invoked from
    ``optimizers.WeightConditioning._step``)."""
    import numpy as np

    # Rows have norms 5, 10 and sqrt(2); a 1-D bias sits alongside them.
    w = ivy.array([[3.0, 4.0], [6.0, 8.0], [1.0, 1.0]])
    b = ivy.array([1.0, 2.0])
    v = ivy.Container({"w": w, "b": b})
    # Zero gradients + tiny lr => the gradient-descent update is identity, so
    # any change to ``w`` comes purely from the equilibration step.
    grads = ivy.Container({"w": ivy.zeros_like(w), "b": ivy.zeros_like(b)})

    out = WeightConditioning(lr=1e-3).step(v, grads)

    row_norms = np.asarray(ivy.vector_norm(out["w"], axis=1).to_numpy())
    assert row_norms == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)

    bias = np.asarray(out["b"].to_numpy())
    assert bias == pytest.approx([1.0, 2.0], abs=1e-6)


def test_step_narrows_singular_value_spread(any_backend):
    """Equilibration reduces the condition number of an ill-conditioned
    weight matrix -- the core thesis of weight conditioning."""
    import numpy as np

    w = ivy.array([[10.0, 0.0], [0.0, 1.0]])  # condition number 10
    v = ivy.Container({"w": w})
    grads = ivy.Container({"w": ivy.zeros_like(w)})

    out = WeightConditioning(lr=1e-3).step(v, grads)

    cond_before = np.linalg.cond(np.asarray(w.to_numpy()))
    cond_after = np.linalg.cond(np.asarray(out["w"].to_numpy()))
    assert cond_after < cond_before
