"""Integration tests for the ``TransferFunction`` state-space layer.

These tests exercise the layer through the public ``Module`` call interface of
the existing ``ivy.stateful.layers`` module (the call site), verifying the
tensor-in/tensor-out contract, linearity, causality and the feed-through
reduction of the state-free FFT inference.
"""

# global
import numpy as np

# local
import ivy
from ivy.stateful.layers import TransferFunction


def _make_layer(
    in_ch, out_ch, state, pole_logit=-3.0, feedthrough=None, recurrent=True
):
    """Build a layer with deterministic, stable parameters via the public ``v``
    constructor argument."""
    s = state
    feedthrough = (
        ivy.zeros((out_ch, in_ch)) if feedthrough is None else ivy.array(feedthrough)
    )
    output_mix = ivy.ones((out_ch, s)) if recurrent else ivy.zeros((out_ch, s))
    v = {
        "pole_logit": ivy.full((s,), pole_logit),
        "pole_angle": ivy.zeros((s,)),
        "input_mix": ivy.ones((s, in_ch)),
        "output_mix": output_mix,
        "feedthrough": feedthrough,
    }
    return TransferFunction(in_ch, out_ch, state, v=v)


def test_transfer_function_shape_contract():
    # Batched and unbatched sequences keep the channel-last tensor contract.
    layer = TransferFunction(3, 2, 4)
    batched = ivy.array(np.random.randn(5, 7, 3).astype("float64"))
    unbatched = ivy.array(np.random.randn(7, 3).astype("float64"))
    assert layer(batched).shape == (5, 7, 2)
    assert layer(unbatched).shape == (7, 2)


def test_transfer_function_is_linear():
    # A state-space layer is a linear time-invariant filter: superposition holds.
    layer = TransferFunction(2, 3, 4)
    x1 = ivy.array(np.random.randn(2, 6, 2).astype("float64"))
    x2 = ivy.array(np.random.randn(2, 6, 2).astype("float64"))
    a, b = 0.7, -1.3
    mixed = layer(ivy.add(ivy.multiply(x1, a), ivy.multiply(x2, b)))
    stacked = ivy.add(ivy.multiply(layer(x1), a), ivy.multiply(layer(x2), b))
    assert np.allclose(mixed.to_numpy(), stacked.to_numpy(), atol=1e-10)


def test_transfer_function_feedthrough_linear_map():
    # With no recurrence, the layer reduces to the per-timestep feed-through map.
    d = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 0.0]])
    layer = _make_layer(2, 3, 2, feedthrough=d, recurrent=False)
    x = ivy.array(np.random.randn(4, 6, 2).astype("float64"))
    expected = np.einsum("oi,bni->bno", d, x.to_numpy())
    assert np.allclose(layer(x).to_numpy(), expected, atol=1e-10)


def test_transfer_function_is_causal():
    # State-free inference is still causal: an impulse cannot affect past output.
    n, t0, in_ch, out_ch = 8, 4, 3, 5
    layer = _make_layer(in_ch, out_ch, 3)
    x = np.zeros((1, n, in_ch))
    x[0, t0, :] = 1.0
    y = layer(ivy.array(x)).to_numpy()
    assert np.abs(y[0, :t0]).max() < 1e-10
    assert np.abs(y[0, t0:]).max() > 0.0
