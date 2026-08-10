# For Review
"""Weight conditioning: row-equilibrate weight matrices to reduce their
condition number.

The core mechanism here is the weight-conditioning normalization of
Zhang et al., "Weight Conditioning for Smooth Optimization of Neural
Networks" (ECCV 2024, arXiv:2409.03424). For a 2-D weight matrix ``W``
each row is scaled to unit L2 norm, i.e. ``W`` is replaced by
``diag(1 / ||W_i:||_2) @ W``. By Van Der Sluis' theorem this diagonal
preconditioner is optimal amongst diagonal preconditioners for reducing
the condition number of ``W``; narrowing the gap between the smallest and
largest singular values smooths the loss landscape and improves the
convergence of gradient-based iterative solvers.

The paper applies the equilibration in the forward pass while keeping
``W`` trainable. This module exposes the same equilibration as a
standalone, parameter-free operator so it can be reused as an
optimizer-step preconditioner -- see
:class:`ivy.stateful.optimizers.WeightConditioning`.
"""

# global
import ivy


def _equilibrate_leaf(w, eps):
    # Only 2-D matrices are conditioned; biases (1-D) and other tensors are
    # returned unchanged so a full variables container can be passed through.
    if not ivy.is_array(w) or len(ivy.shape(w)) != 2:
        return w
    row_norms = ivy.vector_norm(w, axis=1, keepdims=True)
    row_norms = ivy.maximum(row_norms, eps)
    return w / row_norms


def weight_conditioning(w, eps=1e-12):
    """Row-equilibirate weight matrices so each row has unit L2 norm.

    For every 2-D leaf ``W`` of ``w`` this returns
    ``diag(1 / ||W_i:||_2) @ W``; non-2-D leaves (biases, etc.) are
    returned unchanged.

    Parameters
    ----------
    w
        Either a single array or an :class:`ivy.Container` of arrays (for
        example a model's trainable variables). Each 2-D leaf is
        row-equilibrated; the structure of ``w`` is preserved.
    eps
        Numerical floor for the row norms, guarding against division by
        zero on a degenerate (all-zero) row. This is an implementation
        safeguard, not a tuning knob -- the underlying equilibration has
        no hyperparameters.

    Returns
    -------
    ret
        ``w`` with every 2-D leaf row-equilibrated, structured identically
        to the input.
    """
    if isinstance(w, ivy.Container):
        return w.cont_map(lambda x, kc: _equilibrate_leaf(x, eps))
    return _equilibrate_leaf(w, eps)
