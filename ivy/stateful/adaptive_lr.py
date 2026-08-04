# For Review
"""Adaptive learning-rate scheduling from overshoot/stagnation signals.

Provides the dynamic learning-rate adjustment used by the :class:`AdamZ`
optimizer: scale the learning rate *down* when a step overshoots and *up* when
training stagnates, then clamp to ``[lr_min, lr_max]`` and clip gradients to a
maximum global norm.

Adapted from AdamZ (Karimi & Bhatt, 2024; arXiv:2411.15375), an Adam variant
that dynamically adjusts the learning rate to curb overshooting and escape
stagnation. This is a **Mode 2 (adapted) port**: the core mechanism -- halve
the learning rate on overshoot, scale it up by the stagnation factor on
stagnation, clamp to ``[lr_min, lr_max]``, and clip gradients to a max norm --
is preserved at full fidelity with the paper's default factors (overshoot
factor ``0.5``, stagnation factor ``1.2``, max grad norm ``1.0``,
``lr_min=1e-7``, ``lr_max=1``). The paper detects overshoot/stagnation from the
*loss trajectory*; ivy's ``Optimizer._step(v, grads)`` contract does not receive
the loss, so the detection signal is substituted with gradient-based proxies
that approximate the same condition:

* **overshoot** -- successive gradient directions oppose
  (``cosine(prev, cur) < overshoot_threshold``), i.e. the optimizer over-shot a
  valley and is correcting back.
* **stagnation** -- the gradient norm drops below ``stagnation_threshold`` for
  ``patience`` consecutive steps, i.e. updates have become negligible.
"""

# global
import ivy


# Helpers #
# ------- #


def _flat(x):
    """Flatten an ivy array or a (nested) Container of arrays to one vector.

    Container leaves are visited in a stable (key-sorted) order so two
    containers with identical structure produce aligned vectors.
    """
    if hasattr(x, "cont_to_flat_list"):
        leaves = x.cont_to_flat_list()
        return ivy.concat([ivy.reshape(leaf, (-1,)) for leaf in leaves])
    return ivy.reshape(x, (-1,))


def global_grad_norm(grads):
    """Global L2 norm across every leaf of a gradient Container (or array)."""
    return ivy.vector_norm(_flat(grads))


def gradient_cosine(prev_grads, grads):
    """Cosine similarity between two gradient states.

    Returns ``+1`` when successive gradients agree (smooth progress) and ``-1``
    when they oppose (the optimizer over-shot and is pulling back) -- the
    overshoot signal.
    """
    a = _flat(prev_grads)
    b = _flat(grads)
    denom = ivy.vector_norm(a) * ivy.vector_norm(b) + 1e-12
    return ivy.sum(a * b) / denom


def clip_gradients(grads, max_norm):
    """Globally clip gradients so their norm does not exceed ``max_norm``.

    A no-op when ``max_norm`` is ``None``.
    """
    if max_norm is None:
        return grads
    norm = global_grad_norm(grads)
    scale = ivy.minimum(1.0, max_norm / (norm + 1e-12))
    return grads * scale


def _clamp(value, low, high):
    return min(max(value, low), high)


# Scheduling #
# ---------- #


def adjust_learning_rate(
    lr,
    prev_grads,
    grads,
    stagnation_count,
    overshoot_factor=0.5,
    stagnation_factor=1.2,
    overshoot_threshold=0.0,
    stagnation_threshold=1e-6,
    patience=10,
    lr_min=1e-7,
    lr_max=1.0,
):
    """Return ``(new_lr, new_stagnation_count)`` from AdamZ overshoot/stagnation.

    Applies the AdamZ dynamic learning-rate rule using gradient-based proxies
    for the paper's loss-trajectory detection (see module docstring):

    * **overshoot**: if a previous gradient exists and its direction opposes the
      current one (``cosine < overshoot_threshold``), multiply ``lr`` by
      ``overshoot_factor`` and reset the stagnation counter.
    * **stagnation**: otherwise, if the gradient norm stays below
      ``stagnation_threshold`` for ``patience`` consecutive steps, multiply
      ``lr`` by ``stagnation_factor`` and reset the counter.

    The learning rate is clamped to ``[lr_min, lr_max]`` after either
    adjustment.

    Parameters
    ----------
    lr
        Current learning rate.
    prev_grads
        Gradients from the previous step, or ``None`` on the first step.
    grads
        Current gradients.
    stagnation_count
        Consecutive low-gradient steps accumulated so far.
    overshoot_factor
        Factor to multiply ``lr`` by on overshoot. Default ``0.5`` (halve).
    stagnation_factor
        Factor to multiply ``lr`` by on stagnation. Default ``1.2``.
    overshoot_threshold
        Cosine-similarity threshold below which a step counts as an overshoot.
        Default ``0.0`` (opposing directions).
    stagnation_threshold
        Gradient-norm threshold below which a step counts toward stagnation.
        Default ``1e-6``.
    patience
        Consecutive stagnating steps before the learning rate is grown.
        Default ``10``.
    lr_min, lr_max
        Inclusive clamp bounds on the learning rate.
    """
    lr = float(lr)
    if prev_grads is not None:
        cos = float(gradient_cosine(prev_grads, grads))
        if cos < overshoot_threshold:
            lr = _clamp(lr * overshoot_factor, lr_min, lr_max)
            return lr, 0

    if float(global_grad_norm(grads)) < stagnation_threshold:
        stagnation_count += 1
    else:
        stagnation_count = 0

    if stagnation_count >= patience:
        lr *= stagnation_factor
        stagnation_count = 0

    return _clamp(lr, lr_min, lr_max), stagnation_count
