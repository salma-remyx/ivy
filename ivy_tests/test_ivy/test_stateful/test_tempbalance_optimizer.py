"""Tests for the TempBalance layer-wise learning-rate optimizer."""

# global
import numpy as np

# local
import ivy
from ivy.stateful.optimizers import TempBalance


def _mean_abs_update(before, after):
    return float(ivy.to_numpy(ivy.mean(ivy.abs(before - after))))


def test_tempbalance_assigns_larger_step_to_undertrained_layer():
    # TempBalance should transfer learning-rate budget toward undertrained
    # layers (larger PL Alpha) and away from overtrained ones (smaller PL Alpha).
    ivy.set_backend("numpy")
    try:
        rng = np.random.RandomState(0)
        # Undertrained: a random-like, well-spread spectrum -> larger PL Alpha.
        undertrained = ivy.array(rng.randn(6, 6))
        # Overtrained: one dominant spike -> smaller PL Alpha.
        overtrained = ivy.array(np.diag([10.0, 0.1, 0.1, 0.1, 0.1, 0.1]))
        v = ivy.Container({"l0": {"w": undertrained}, "l1": {"w": overtrained}})
        grads = ivy.Container(
            {"l0": {"w": ivy.ones((6, 6))}, "l1": {"w": ivy.ones((6, 6))}}
        )

        optimizer = TempBalance(lr=0.1)
        updated = optimizer.step(v, grads)

        move_undertrained = _mean_abs_update(v.l0.w, updated.l0.w)
        move_overtrained = _mean_abs_update(v.l1.w, updated.l1.w)

        # Eq. 2 maps the largest-PL-Alpha layer to s2 and the smallest to s1.
        assert abs(move_undertrained - 0.1 * 1.5) < 1e-5
        assert abs(move_overtrained - 0.1 * 0.5) < 1e-5
        assert move_undertrained > move_overtrained
    finally:
        ivy.previous_backend()


def test_tempbalance_multiplier_map_is_scale_invariant():
    # Rescaling all weights by a constant must not change the (scale-free)
    # per-layer multiplier map, so the resulting step sizes are identical.
    ivy.set_backend("numpy")
    try:
        rng = np.random.RandomState(1)
        v = ivy.Container(
            {
                "a": {"w": ivy.array(rng.randn(5, 5))},
                "b": {"w": ivy.array(np.diag([9.0, 0.2, 0.2, 0.2, 0.2]))},
            }
        )
        grads = ivy.Container(
            {"a": {"w": ivy.ones((5, 5))}, "b": {"w": ivy.ones((5, 5))}}
        )

        move_a = _mean_abs_update(v.a.w, TempBalance(lr=0.1).step(v, grads).a.w)
        move_b = _mean_abs_update(v.b.w, TempBalance(lr=0.1).step(v, grads).b.w)
        scaled = v * 10.0
        move_a_scaled = _mean_abs_update(
            scaled.a.w, TempBalance(lr=0.1).step(scaled, grads).a.w
        )
        move_b_scaled = _mean_abs_update(
            scaled.b.w, TempBalance(lr=0.1).step(scaled, grads).b.w
        )

        assert abs(move_a - move_a_scaled) < 1e-6
        assert abs(move_b - move_b_scaled) < 1e-6
    finally:
        ivy.previous_backend()
