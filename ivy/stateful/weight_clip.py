# For Review
"""Post-update weight clipping for Ivy optimizers.

Implements the element-wise weight-clipping rule of "Weight Clipping for Deep
Continual and Reinforcement Learning" (Kumar et al., arXiv:2407.01704). The
paper observes that many deep continual- and reinforcement-learning failures
track the growth of weight magnitude, and proposes a single remedy applied
*on top of* any existing optimizer: after each gradient step, project every
updated parameter back into the closed interval ``[-clip_value, clip_value]``.

Because the clip is a post-update projection, it neither replaces the base
optimizer nor alters the model architecture -- the property the paper credits
for its easy adoption across systems. ``WeightClip`` realises that contract by
wrapping any other Ivy optimizer and clipping its ``_step`` output; the
``clip_weights`` helper exposes the same projection as a standalone utility for
callers that apply it manually.
"""

# global
from typing import Union, Callable

# local
import ivy
from ivy.stateful.optimizers import Optimizer


def _resolve_bound(bound: Union[float, Callable]) -> float:
    """Resolve a clip bound that may be supplied as a constant or a callable."""
    return bound if isinstance(bound, float) else bound()


def clip_weights(
    v: ivy.Container,
    clip_value: Union[float, Callable] = 1.0,
) -> ivy.Container:
    """Project a nested variables container element-wise into ``[-c, c]``.

    This is the post-update weight-clipping rule: each parameter value is
    clamped to the symmetric range ``[-clip_value, clip_value]``. It is the
    primitive ``WeightClip`` applies after every optimizer step, exposed on its
    own for manual use.

    Parameters
    ----------
    v
        Nested variables container to clip.
    clip_value
        Element-wise bound ``c``. Values below ``-c`` are raised to ``-c`` and
        values above ``c`` are lowered to ``c``. Default is ``1.0``.

    Returns
    -------
    ret
        A new nested variables container with every leaf clamped to
        ``[-clip_value, clip_value]``.
    """
    c = _resolve_bound(clip_value)
    return ivy.clip(v, -c, c)


class WeightClip(Optimizer):
    def __init__(
        self,
        base_optimizer: Optimizer,
        clip_value: Union[float, Callable] = 1.0,
    ):
        """Construct a weight-clipping optimizer that wraps a base optimizer.

        The wrapped optimizer's gradient step is performed unchanged; its
        updated variables are then projected element-wise into
        ``[-clip_value, clip_value]`` before being returned. The base
        optimizer's internal state (e.g. Adam's moment estimates and step
        counter) advances exactly as it would without clipping, so the clip
        layers on top of the existing learning system rather than replacing it.

        Parameters
        ----------
        base_optimizer
            Any Ivy optimizer (``SGD``, ``Adam``, ``LAMB``, ...) whose step is
            performed before clipping.
        clip_value
            Element-wise bound ``c``; updated weights are clamped to
            ``[-c, c]`` after each step. Default is ``1.0``.
        """
        Optimizer.__init__(
            self,
            base_optimizer._lr,
            base_optimizer._inplace,
            base_optimizer._stop_gradients,
            False,
            base_optimizer._trace_on_next_step,
            device=base_optimizer._dev,
        )
        self._base = base_optimizer
        self._clip_value = clip_value

    # Custom Step

    def _step(self, v: ivy.Container, grads: ivy.Container):
        """Update ``v`` by the base optimizer's step, then clip it.

        Parameters
        ----------
        v
            Nested variables to update.
        grads
            Nested gradients to update.

        Returns
        -------
        ret
            The base optimizer's updated variables, projected element-wise into
            ``[-clip_value, clip_value]``.
        """
        new_v = self._base.step(v, grads)
        c = _resolve_bound(self._clip_value)
        return ivy.clip(new_v, -c, c)

    def set_state(self, state: ivy.Container):
        """Set the state of the underlying base optimizer.

        Parameters
        ----------
        state
            Nested state to update.
        """
        self._base.set_state(state)

    @property
    def state(self):
        return self._base.state
