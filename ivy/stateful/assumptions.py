# For Review
"""Empirical tests of the assumptions used to analyse optimizer convergence.

The metrics here follow "Reevaluating Theoretical Analysis Methods for
Optimization in Deep Learning" (Tran, Zhang & Cutkosky, 2024,
arXiv:2407.01825), which measures whether the quantities that standard
convergence proofs take for granted -- local convexity, L-smoothness, and
the descent lemma -- actually hold along a real optimization trajectory,
rather than assuming they do.

The tracker is a passive observer: it never touches the parameters or the
gradients it is handed, it only measures them.
"""

# global
from typing import Optional

# local
import ivy


def _flatten(v: ivy.Container) -> ivy.Array:
    """Concatenate every leaf of a nested container into one flat array."""
    leaves = [ivy.reshape(sub, (-1,)) for _, sub in v.cont_to_iterator()]
    return ivy.concat(leaves, axis=0)


class AssumptionTracker:
    def __init__(self, beta: float = 0.99, epsilon: float = 1e-12):
        """Record the trajectory-level quantities that convergence analyses
        rely on, so they can be checked against the optimizer's actual
        behaviour instead of being assumed.

        Parameters
        ----------
        beta
            Decay factor for the exponential moving averages ``exp_smooth``
            and ``exp_update_corr``. Default is ``0.99``.
        epsilon
            Numerical floor used when dividing by a parameter distance,
            so a repeated iterate yields a smoothness of ``0`` rather than
            a division by zero. Default is ``1e-12``.
        """
        self._beta = beta
        self._epsilon = epsilon
        self._steps = 0
        self._prev_v = None
        self._prev_grad = None
        self._pending_delta = None
        self._pending_grad = None
        self._max_smooth = 0.0
        self._exp_smooth = 0.0
        self._exp_update_corr = 0.0
        self._sum_update_corr = 0.0

    # Private #
    # --------#

    def _smoothness(self, grad: ivy.Array, prev_grad: ivy.Array, dist: ivy.Array):
        """Ratio ||grad - prev_grad|| / ||v - prev_v||, the local Lipschitz
        constant of the gradient that the L-smoothness assumption bounds.
        """
        return ivy.vector_norm(grad - prev_grad) / ivy.maximum(dist, self._epsilon)

    def _scalar(self, name: str) -> Optional[float]:
        value = self.__dict__.get(name)
        return None if value is None else float(ivy.to_scalar(value))

    # Given #

    def record(self, v: ivy.Container, grads: ivy.Container, new_v: ivy.Container):
        """Measure one update ``v -> new_v`` taken with gradient ``grads``.

        Called once per optimizer step with the parameters before and after
        the update. The first call only stores a reference point, since
        every metric needs at least two iterates.

        Parameters
        ----------
        v
            Variables before the update step.
        grads
            Gradients the update was computed from.
        new_v
            Variables after the update step.
        """
        v_flat = _flatten(v)
        new_v_flat = _flatten(new_v)
        grad_flat = _flatten(grads)
        delta = new_v_flat - v_flat

        if self._prev_v is not None:
            prev_v_flat = _flatten(self._prev_v)
            prev_grad_flat = _flatten(self._prev_grad)

            # Eq. 4: instantaneous smoothness, the local Lipschitz constant of
            # the gradient that L-smoothness bounds above by L. The convexity
            # gap (paper Eq. 2) needs the stochastic loss f(x_t, z_t), which
            # the optimizer loop never evaluates, so it lives outside this
            # tracker rather than being approximated here.
            self._inst_smooth = self._smoothness(
                grad_flat, prev_grad_flat, ivy.vector_norm(v_flat - prev_v_flat)
            )
            self._max_smooth = max(self._max_smooth, self._inst_smooth)
            self._exp_smooth = (
                self._beta * self._exp_smooth + (1 - self._beta) * self._inst_smooth
            )

        if self._pending_delta is not None and self._pending_grad is not None:
            # Eq. 9: update correlation <grad(x_{t+1}), x_{t+1} - x_t>, the
            # quantity the descent lemma needs to be negative. The gradient at
            # the *new* iterate is unavailable inside ``step``, so it is paired
            # with the incoming gradient of the next call, which the paper
            # reports is an unbiased estimator of the same inner product.
            self._inst_update_corr = ivy.dot(grad_flat, self._pending_delta)
            self._sum_update_corr += self._inst_update_corr
            self._exp_update_corr = (
                self._beta * self._exp_update_corr
                + (1 - self._beta) * self._inst_update_corr
            )

        self._prev_v = v
        self._prev_grad = grads
        self._pending_delta = delta
        self._pending_grad = grad_flat
        self._steps += 1

    # Public #
    # -------#

    @property
    def steps(self) -> int:
        """Number of recorded updates."""
        return self._steps

    @property
    def inst_smooth(self) -> Optional[float]:
        """Most recent instantaneous smoothness (paper Eq. 4), or ``None``
        before two iterates have been seen."""
        return self._scalar("_inst_smooth")

    @property
    def max_smooth(self) -> Optional[float]:
        """Running maximum of the instantaneous smoothness. If the objective
        were L-smooth this would settle at L; a value that keeps growing over
        training is the edge-of-stability signature the paper reports."""
        return self._scalar("_max_smooth")

    @property
    def exp_smooth(self) -> Optional[float]:
        """Exponentially weighted instantaneous smoothness, which the paper
        found tracks Hessian sharpness closely at a fraction of its cost."""
        return self._scalar("_exp_smooth")

    @property
    def inst_update_corr(self) -> Optional[float]:
        """Most recent update correlation ``<grad(x_new), x_new - x_old>``
        (paper Eq. 9). The descent lemma needs this negative; the paper finds
        it positive on average in almost every deep learning setting."""
        return self._scalar("_inst_update_corr")

    @property
    def exp_update_corr(self) -> Optional[float]:
        """Exponentially weighted update correlation."""
        return self._scalar("_exp_update_corr")

    @property
    def sum_update_corr(self) -> Optional[float]:
        """Cumulative update correlation. The identity the paper validates --
        that this tracks the cumulative loss change -- is what makes it a
        usable stand-in for the descent-lemma quantity."""
        return self._scalar("_sum_update_corr")
