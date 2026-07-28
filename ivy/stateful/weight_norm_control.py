"""Weight-norm-control decay for optimizers.

Implements the weight-norm-control generalization of weight decay described in
"Weight Norm Control" (arxiv:2311.11446). Ordinary (decoupled) weight decay
shrinks every parameter toward the zero vector, i.e. it is weight-norm control
with a target L2 norm of ``0``. Allowing a non-zero ``target_norm`` instead pulls
each parameter tensor toward that norm while leaving its direction untouched,
which the paper argues can be less suboptimal than decaying to zero.

``target_norm = 0`` recovers plain weight decay exactly, so an optimizer built on
this term (e.g. :class:`ivy.stateful.optimizers.AdamWN`) is a strict
generalization of the corresponding weight-decay optimizer (e.g.
:class:`ivy.stateful.optimizers.AdamW`).
"""

# local
import ivy


def weight_norm_control_term(v, weight_decay, target_norm, epsilon=1e-12):
    """Return the weight-norm-control gradient term to add to ``grads``.

    For each parameter tensor the returned term is
    ``weight_decay * (v - target_norm * v / (||v|| + epsilon))``: decoupled decay
    toward ``target_norm`` rather than toward zero. With ``target_norm = 0`` this
    reduces to the usual ``weight_decay * v`` weight-decay term, so the caller
    behaves identically to a plain weight-decay optimizer.

    Parameters
    ----------
    v
        Nested variables container (or array) whose L2 norms are controlled. The
        norm is computed independently per leaf tensor, so each parameter is
        pulled toward its own ``target_norm``.
    weight_decay
        Weight-decay / norm-control coefficient.
    target_norm
        Target L2 norm each parameter tensor is pulled toward.
    epsilon
        Small constant stabilizing the norm in the denominator.

    Returns
    -------
    ret
        Container (or array) of the same shape as ``v`` to be added to the
        gradients before the base optimizer step.
    """
    if target_norm == 0:
        return weight_decay * v
    norm = ivy.vector_norm(v)
    return weight_decay * (v - target_norm * v / (norm + epsilon))
