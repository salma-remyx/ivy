"""Collection of tests for the optimizer assumption tracker."""

# global
import pytest

# local
import ivy
from ivy.stateful import AssumptionTracker


def _quadratic_grad(ivy_backend, v, curvature):
    """Gradient of 0.5 * curvature * x**2, an L-smooth function with L equal
    to ``curvature`` and minimizer at zero."""
    return curvature * v


@pytest.mark.parametrize("curvature", [0.5, 2.0, 7.5])
def test_smoothness_recovers_curvature(curvature, backend_fw):
    with ivy.utils.backend.ContextManager(backend_fw) as ivy_backend:
        optimizer = ivy.stateful.optimizers.SGD(lr=0.1)
        tracker = AssumptionTracker()
        v = ivy_backend.Container({"w": ivy_backend.array([3.0, -1.5])})
        for _ in range(6):
            grads = _quadratic_grad(ivy_backend, v, curvature)
            v = optimizer.step(v, grads, tracker=tracker)

        # For a quadratic, the instantaneous smoothness is exactly L at every
        # step, and it never grows past it.
        assert tracker.steps == 6
        assert tracker.inst_smooth is not None
        assert abs(tracker.inst_smooth - curvature) < 1e-4
        assert tracker.max_smooth <= curvature + 1e-4


def test_update_corr_negative_when_descending(backend_fw):
    with ivy.utils.backend.ContextManager(backend_fw) as ivy_backend:
        optimizer = ivy.stateful.optimizers.SGD(lr=0.1)
        tracker = AssumptionTracker()
        v = ivy_backend.Container({"w": ivy_backend.array([3.0])})
        for _ in range(5):
            grads = _quadratic_grad(ivy_backend, v, 2.0)
            v = optimizer.step(v, grads, tracker=tracker)

        # Gradient descent on a convex objective satisfies the descent lemma,
        # so the update correlation must be negative at every measured step.
        assert tracker.inst_update_corr < 0
        assert tracker.sum_update_corr < 0


def test_smoothness_grows_without_lipschitz_bound(backend_fw):
    with ivy.utils.backend.ContextManager(backend_fw) as ivy_backend:
        optimizer = ivy.stateful.optimizers.SGD(lr=0.05)
        tracker = AssumptionTracker()
        v = ivy_backend.Container({"w": ivy_backend.array([1.0])})
        smoothness = []
        for _ in range(20):
            w = v["w"]
            # Gradient of |x|**1.5, which is not Lipschitz near the minimizer.
            grads = ivy_backend.Container(
                {"w": 1.5 * ivy_backend.sign(w) * ivy_backend.abs(w) ** 0.5}
            )
            v = optimizer.step(v, grads, tracker=tracker)
            if tracker.inst_smooth is not None:
                smoothness.append(tracker.inst_smooth)

        # No single L bounds this gradient, so the measured smoothness keeps
        # rising over training instead of settling at a constant.
        assert smoothness[-1] > 2.0 * smoothness[0]
        assert tracker.max_smooth > 1.0


def test_step_is_unchanged_without_tracker(backend_fw):
    with ivy.utils.backend.ContextManager(backend_fw) as ivy_backend:
        v0 = ivy_backend.Container({"w": ivy_backend.array([3.0, -1.5])})
        grads = _quadratic_grad(ivy_backend, v0, 2.0)

        with_tracker = ivy.stateful.optimizers.SGD(lr=0.1).step(
            v0, grads, tracker=AssumptionTracker()
        )
        without_tracker = ivy.stateful.optimizers.SGD(lr=0.1).step(v0, grads)

        # The tracker only observes, it must never perturb the update.
        assert ivy_backend.all_equal(
            ivy_backend.equal(with_tracker["w"], without_tracker["w"])
        )
