"""HT-SR layer-temperature helpers for layer-wise learning-rate scaling.

Provides the per-layer "temperature" estimator and the layer-wise
learning-rate assignment used by the
:class:`~ivy.stateful.optimizers.TempBalance` optimizer. Adapted from
Zhou et al., "Temperature Balancing, Layer-wise Weight Analysis, and Neural
Network Training" (NeurIPS 2023, arXiv:2312.00359), which balances learning
rates across layers using Heavy-Tailed Self-Regularization (HT-SR) theory.
"""

# global
import math

# local
import ivy


def pl_alpha_hill(weight):
    """Estimate the HT-SR PL Alpha of a 2-D weight matrix via the Hill estimator.

    Forms the empirical spectral distribution (ESD) of ``weight`` -- the
    eigenvalues of ``W^T W`` -- and fits its power-law tail with the Hill
    estimator (Hill, 1975), using ``k = n // 2`` of the ``n`` eigenvalues, as
    in TempBalance. A *larger* value indicates a layer whose spectrum is less
    heavy-tailed (relatively undertrained); a *smaller* value indicates a more
    heavy-tailed, overtrained layer.

    The estimator is only used to *order* layers, never for its absolute value
    (see :func:`layerwise_lr_multipliers`), so it is evaluated outside the
    training graph as a detached schedule quantity.

    Parameters
    ----------
    weight
        2-D weight matrix.

    Returns
    -------
    ret
        The estimated PL Alpha as a Python float, or ``None`` when the tail
        cannot be estimated (non-2-D input, too few eigenvalues, or a
        degenerate spectrum).
    """
    if getattr(weight, "ndim", None) != 2:
        return None
    # Gram matrix of the smaller side; shares the nonzero spectrum of W^T W.
    if int(weight.shape[1]) <= int(weight.shape[0]):
        gram = ivy.matmul(ivy.swapaxes(weight, -1, -2), weight)
    else:
        gram = ivy.matmul(weight, ivy.swapaxes(weight, -1, -2))
    eigvals = ivy.to_numpy(ivy.abs(ivy.eigh(gram)[0]))
    n = eigvals.shape[0]
    k = n // 2
    if k < 1:
        return None
    threshold = float(eigvals[n - k - 1])
    if threshold <= 0.0:
        return None
    log_ratio_sum = 0.0
    for value in eigvals[n - k :]:
        ratio = float(value) / threshold
        if ratio <= 0.0:
            return None
        log_ratio_sum += math.log(ratio)
    if log_ratio_sum <= 0.0:
        return None
    return 1.0 + k / log_ratio_sum


def layerwise_lr_multipliers(v, s1=0.5, s2=1.5):
    """Assign per-layer learning-rate multipliers from HT-SR temperatures.

    Implements the TempBalance assignment (Eq. 2 of the paper): each 2-D
    weight matrix receives a multiplier in ``[s1, s2]`` that linearly maps its
    PL Alpha between the layer-wise minimum and maximum, so relatively
    undertrained layers (larger PL Alpha) get a larger multiplier and
    overtrained layers get a smaller one. Non-matrix parameters (e.g. biases)
    receive ``1.0``. The map is invariant to any affine rescaling of the PL
    Alpha, so the absolute estimator scale is irrelevant.

    Parameters
    ----------
    v
        Nested variables container.
    s1
        Minimum multiplier. Default is ``0.5``.
    s2
        Maximum multiplier. Default is ``1.5``.

    Returns
    -------
    ret
        Container matching ``v`` with a per-leaf multiplier (float).
    """
    leaves = list(v.cont_to_iterator())
    alphas = []
    matrix_indices = []
    for index, (_, value) in enumerate(leaves):
        alpha = pl_alpha_hill(value)
        if alpha is not None and math.isfinite(alpha):
            alphas.append(alpha)
            matrix_indices.append(index)
    multipliers = [1.0] * len(leaves)
    if len(alphas) >= 2:
        alpha_min = min(alphas)
        alpha_max = max(alphas)
        spread = alpha_max - alpha_min
        if spread > 0.0:
            for slot, index in enumerate(matrix_indices):
                alpha = alphas[slot]
                multipliers[index] = s1 + (s2 - s1) * (alpha - alpha_min) / spread
    # ``cont_from_flat_list`` consumes the list in place, so pass a copy.
    return v.cont_from_flat_list(list(multipliers))
