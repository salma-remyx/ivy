"""Backend-agnostic spectral analysis of weight containers.

Adapted from "Approaching Deep Learning through the Spectral Dynamics of
Weights" (Yunis et al., arXiv:2408.11804), which studies the singular values
of weight matrices during optimization and shows that their dynamics carry
more signal than weight norms alone: optimization drives singular values
apart (rich-get-richer), weight decay amplifies that spread, and the bulk /
tail of the spectrum separates generalizing networks from memorizing ones.

The utilities here expose that signal for any Ivy container of parameters,
so it can be inspected uniformly across PyTorch, TensorFlow, JAX and NumPy
backends.
"""

# global
from typing import Dict, Optional, Union

# local
import ivy

# Minimum number of singular values a matrix must have before stable rank is
# meaningful. Below this the "bulk" of the spectrum is empty.
_MIN_BULK_SIZE = 4


def _singular_values(w) -> ivy.Array:
    """Return the descending singular values of ``w``, flattening any leading
    batch dimensions so convolution kernels ``(..., H, W)`` are treated as a
    stack of matrices.
    """
    w = ivy.asarray(w)
    shape = ivy.shape(w, as_array=True)
    if len(shape) < 2:
        return ivy.zeros((0,), dtype=ivy.dtype(w))
    if len(shape) > 2:
        # (... , M, N) -> (prod(leading), M, N)
        w = ivy.reshape(w, (-1, int(shape[-2]), int(shape[-1])))
    ret = ivy.svd(w, compute_uv=False).S
    return ivy.sort(ivy.asarray(ret), descending=True)


def _stablize(values: ivy.Array, eps: float) -> ivy.Array:
    """Clamp values to at least ``eps`` so ratios stay finite."""
    return ivy.maximum(values, ivy.asarray(eps, dtype=ivy.dtype(values)))


def spectral_summary(w, eps: float = 1e-12) -> Dict[str, Union[ivy.Array, int, float]]:
    """Summarize the singular-value spectrum of a single weight matrix.

    The tracked statistics follow the spectral-dynamics framing of the paper:
    the spectral norm (largest singular value) tracks the head of the
    spectrum that gradient descent grows, while the stable rank and the tail
    mass track the bulk that lags behind.

    Parameters
    ----------
    w
        Weight matrix, or array with at least two trailing matrix
        dimensions (leading dimensions are treated as a batch of matrices).
    eps
        Numerical floor used when normalizing by the spectral norm.
        Default is ``1e-12``.

    Returns
    -------
    ret
        Dict with keys ``s`` (descending singular values), ``sigma_max``,
        ``fro_norm``, ``stable_rank`` and ``tail_mass_ratio``.
    """
    s = _singular_values(w)
    if ivy.shape(s, as_array=True)[0] == 0:
        return {
            "s": s,
            "sigma_max": ivy.asarray(0.0, dtype=ivy.dtype(s)),
            "fro_norm": ivy.asarray(0.0, dtype=ivy.dtype(s)),
            "stable_rank": ivy.asarray(0.0, dtype=ivy.dtype(s)),
            "tail_mass_ratio": ivy.asarray(0.0, dtype=ivy.dtype(s)),
        }
    sigma_max = ivy.asarray(s[0])
    fro_norm = ivy.vector_norm(s)
    stable_rank = ivy.asarray(fro_norm**2) / _stablize(ivy.asarray(sigma_max**2), eps)
    if int(ivy.shape(s, as_array=True)[0]) >= _MIN_BULK_SIZE:
        # Energy carried by all but the leading singular value, relative to
        # the whole spectrum: high means a flat bulk, low means the head
        # dominates.
        tail = fro_norm**2 - ivy.asarray(sigma_max**2)
        tail_mass_ratio = tail / _stablize(ivy.asarray(fro_norm**2), eps)
    else:
        tail_mass_ratio = ivy.asarray(float("nan"), dtype=ivy.dtype(s))
    return {
        "s": s,
        "sigma_max": sigma_max,
        "fro_norm": fro_norm,
        "stable_rank": stable_rank,
        "tail_mass_ratio": tail_mass_ratio,
    }


def container_spectral_summary(v: ivy.Container, eps: float = 1e-12) -> ivy.Container:
    """Summarize the spectra of every matrix parameter in a variables
    container.

    Parameters
    ----------
    v
        Nested variables container, e.g. ``Module.v`` or the ``v`` passed to
        ``Optimizer._step``. One-dimensional leaves (biases, norms) are
        skipped, since they have no spectrum.
    eps
        Numerical floor used when normalizing. Default is ``1e-12``.

    Returns
    -------
    ret
        Container mirroring the structure of ``v``, whose leaves hold the
        ``spectral_summary`` dict of the corresponding matrix parameter.
    """
    # Build a plain nested dict and wrap once at the end: setting a dict leaf
    # with cont_set_at_key_chain would nest inside a previously set dict leaf
    # rather than replace it.
    ret = {}
    for key_chain, value in v.cont_to_iterator():
        shape = ivy.shape(value, as_array=True)
        if len(shape) < 2:
            continue
        node = ret
        keys = key_chain.split("/")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = spectral_summary(value, eps=eps)
    return ivy.Container(ret)


class WeightSpectrumTracker:
    """Record the spectral summary of a variables container over optimizer
    steps.

    The paper's core observation is that these summaries move in
    characteristic ways during training -- the head of the spectrum grows
    faster than the bulk -- and that watching that movement distinguishes
    memorization from generalization earlier than watching loss does. This
    tracker holds the per-step history so a caller can plot or diff it.

    Parameters
    ----------
    top_k
        Number of leading singular values to keep per parameter, in addition
        to the scalar summaries. Default is ``1``.
    eps
        Numerical floor used when normalizing. Default is ``1e-12``.

    Examples
    --------
    >>> tracker = WeightSpectrumTracker()
    >>> opt = ivy.optimizers.Adam(1e-3, track_spectrum=tracker)
    >>> for step in range(n_steps):
    ...     v = opt.step(v, grads)
    >>> tracker.df  # per-step history
    """

    def __init__(self, top_k: int = 1, eps: float = 1e-12):
        self._top_k = top_k
        self._eps = eps
        self._history = []
        self._step_count = 0

    # Private #
    # --------#

    def _record(self, v: Optional[ivy.Container]):
        """Append the spectral summary of ``v`` to the history."""
        if v is None:
            return
        step = self._step_count
        summaries = container_spectral_summary(v, eps=self._eps)
        # Each summarised parameter became a sub-container holding its
        # summary fields, so the parameter key-chains are one level above the
        # flattened leaves.
        param_key_chains = sorted(
            {
                key_chain.rsplit("/", 1)[0]
                for key_chain in summaries.cont_to_iterator_keys()
            }
        )
        for param_key_chain in param_key_chains:
            summary = summaries.cont_at_key_chain(param_key_chain)
            s = ivy.to_numpy(ivy.asarray(summary.s)).ravel()
            entry = {
                "step": step,
                "key_chain": param_key_chain,
                "sigma_max": float(ivy.to_numpy(summary.sigma_max)),
                "fro_norm": float(ivy.to_numpy(summary.fro_norm)),
                "stable_rank": float(ivy.to_numpy(summary.stable_rank)),
            }
            tail = summary.tail_mass_ratio
            entry["tail_mass_ratio"] = (
                float("nan")
                if ivy.isnan(ivy.asarray(tail))
                else float(ivy.to_numpy(tail))
            )
            for i in range(min(self._top_k, s.shape[0])):
                entry[f"s_{i}"] = float(s[i])
            self._history.append(entry)
        self._step_count += 1

    # Public #
    # -------#

    def record(self, v):
        """Record the spectral summary of a variables container or module.

        Parameters
        ----------
        v
            Nested variables container, or an ``ivy.Module`` whose ``v`` is
            analysed.
        """
        self._record(getattr(v, "v", v))

    @property
    def history(self):
        """List of per-step, per-parameter summary dicts."""
        return self._history

    @property
    def df(self):
        """Alias of :attr:`history`, for tabular consumers."""
        return self._history

    def spectrum_drift(self, key_chain: str = "") -> Dict[str, float]:
        """Return the change in each scalar summary between the first and
        last recorded step, averaged over parameters.

        Positive ``sigma_max`` drift with negative ``tail_mass_ratio`` drift
        is the spectral signature the paper associates with optimization
        driving the head of the spectrum away from the bulk.

        Parameters
        ----------
        key_chain
            Restrict the comparison to parameters whose key-chain starts
            with this prefix. Default is ``""`` (all parameters).
        """
        if not self._history:
            return {}
        prefix = key_chain.rstrip("/")
        steps = {}
        for entry in self._history:
            if prefix and not entry["key_chain"].startswith(prefix):
                continue
            steps.setdefault(entry["key_chain"], []).append(entry)
        drift = {}
        for entries in steps.values():
            first, last = entries[0], entries[-1]
            for field in ("sigma_max", "fro_norm", "stable_rank", "tail_mass_ratio"):
                if field in first and field in last:
                    drift[field] = drift.get(field, 0.0) + (last[field] - first[field])
        n = len(steps)
        return {field: value / n for field, value in drift.items()}
