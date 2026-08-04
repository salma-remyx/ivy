"""Collection of Ivy normalization classes."""

# local
import ivy
from ivy.stateful.module import Module
from ivy.stateful.initializers import Zeros, Ones


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape,
        /,
        *,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        new_std: float = 1.0,
        device=None,
        v=None,
        dtype=None,
    ):
        """Class for applying Layer Normalization over a mini-batch of inputs.

        Parameters
        ----------
        normalized_shape
            Trailing shape to applying the normalization to.
        epsilon
            small constant to add to the denominator,
            use global ivy.min_base by default.
        elementwise_affine
            Whether to include learnable affine parameters, default is ``True``.
        new_std
            The standard deviation of the new normalized values. Default is 1.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. (Default value = None)
        v
            the variables for each submodule in the sequence,
            constructed internally by default.
        """
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self._normalized_idxs = [-(i + 1) for i in range(len(normalized_shape))]
        self._epsilon = eps
        self._elementwise_affine = elementwise_affine
        self._new_std = new_std
        self._weight_shape = normalized_shape
        self._bias_shape = normalized_shape
        self._weight_init = Ones()
        self._bias_init = Zeros()
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device=None, dtype=None):
        """Create internal variables for the layer."""
        device = ivy.default(device, self.device)
        dtype = ivy.default(dtype, self.dtype)
        if self._elementwise_affine:
            return {
                "weight": self._weight_init.create_variables(
                    self._weight_shape, device, dtype=dtype
                ),
                "bias": self._bias_init.create_variables(
                    self._bias_shape, device, dtype=dtype
                ),
            }
        return {}

    def _forward(self, inputs):
        """Perform forward pass of the LayerNorm layer.

        Parameters
        ----------
        inputs
            Inputs to process.

        Returns
        -------
        ret
            The outputs following the layer normalization operation.
        """
        return ivy.layer_norm(
            inputs,
            self._normalized_idxs,
            eps=self._epsilon,
            scale=self.v.weight if self._elementwise_affine else None,
            offset=self.v.bias if self._elementwise_affine else None,
            new_std=self._new_std,
        )

    def _extra_repr(self) -> str:
        return (
            f"normalized_idxs={self._normalized_idxs}, epsilon={self._epsilon}, "
            f"elementwise_affine={self._elementwise_affine}, new_std={self._new_std}"
        )


class BatchNorm2D(Module):
    def __init__(
        self,
        num_features,
        /,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        data_format: str = "NSC",
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        v=None,
        dtype=None,
        training=True,
    ):
        """Class for applying Layer Normalization over a mini-batch of inputs.

        Parameters
        ----------
        num_features
            Trailing shape to applying the normalization to.
        epsilon
            small constant to add to the denominator,
            use global ivy.min_base by default.
        data_format
            The ordering of the dimensions in the input, one of "NSC" or "NCS",
            where N is the batch dimension, S represents any number of spatial
            dimensions and C is the channel dimension. Default is "NSC".
        affine
            Whether to include learnable affine parameters, default is ``True``.
        track_running_stats
            is a boolean flag that determines whether
            the running statistics should be updated
            during training in batch normalization.
        momentum
             The value used for the running_mean and running_var computation.
              Default is 0.1.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. (Default value = None)
        v
            the variables for each submodule in the sequence,
            constructed internally by default.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            internal `mean` and `variance` when affine is True.
        """
        self.num_features = num_features
        self._affine = affine
        self.data_format = data_format
        self._epsilon = eps
        self._momentum = momentum
        self._track_running_stats = track_running_stats
        self._weight_shape = num_features
        self._bias_shape = num_features
        self._running_mean_shape = num_features
        self._running_var_shape = num_features
        self._weight_init = Ones()
        self._bias_init = Zeros()
        self._running_mean_init = Zeros()
        self._running_var_init = Ones()
        Module.__init__(self, device=device, v=v, dtype=dtype, training=training)

    def _create_variables(self, device=None, dtype=None):
        """Create internal variables for the layer."""
        device = ivy.default(device, self.device)
        dtype = ivy.default(dtype, self.dtype)
        if self._affine:
            return {
                "b": self._bias_init.create_variables(
                    self._bias_shape, device, dtype=dtype
                ),
                "running_mean": self._running_mean_init.create_variables(
                    self._running_mean_shape, device, dtype=dtype
                ),
                "running_var": self._running_var_init.create_variables(
                    self._running_var_shape, device, dtype=dtype
                ),
                "w": self._weight_init.create_variables(
                    self._weight_shape, device, dtype=dtype
                ),
            }
        return {}

    def _forward(self, inputs):
        """Perform forward pass of the BatchNorm layer.

        Parameters
        ----------
        inputs
            Inputs to process of shape N,C,*.

        Returns
        -------
        ret
            The outputs following the batch normalization operation.
        """
        normalized, running_mean, running_var = ivy.batch_norm(
            inputs,
            self.v.running_mean,
            self.v.running_var,
            eps=self._epsilon,
            momentum=self._momentum,
            data_format=self.data_format,
            training=self.training,
            scale=self.v.w if self._affine else None,
            offset=self.v.b if self._affine else None,
        )
        if self._track_running_stats and self.training:
            self.v.running_mean = running_mean
            self.v.running_var = running_var

        return normalized

    def _extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, affine={self._affine}, "
            f"data_format={self.data_format}, epsilon={self._epsilon} "
            f"momentum={self._momentum}, "
            f"track_running_stats={self._track_running_stats}"
        )


def weight_conditioning(weights, /, *, mode: str = "row", eps: float = 1e-12, ord=2):
    """Condition a weight matrix by diagonal equilibration.

    Scales each row (or column) of ``weights`` by the reciprocal of its
    vector norm, so every row (column) has unit norm. This narrows the gap
    between the smallest and largest singular values, yielding a
    better-conditioned matrix. The preconditioner is recomputed from the
    current ``weights`` on every call; no SVD is required.

    Adapted from *Weight Conditioning for Smooth Optimization of Neural
    Networks* (arXiv:2409.03424), which identifies row equilibration as the
    optimal diagonal preconditioner (Van Der Sluis' theorem) and notes the
    same statements hold for the column and row-column forms.

    Parameters
    ----------
    weights
        Input weight matrix (or a stack of matrices) whose innermost two
        dimensions form the matrices to equilibrate.
    mode
        ``"row"`` scales each row to unit norm (default, the paper's primary
        form), ``"column"`` scales each column, ``"row_column"`` applies both.
    eps
        Floor applied to each norm to avoid division by zero. Default ``1e-12``.
    ord
        Order of the vector norm used for equilibration. Default ``2``.
    """
    if mode == "row":
        out = weights / ivy.maximum(
            ivy.vector_norm(weights, axis=-1, keepdims=True, ord=ord), eps
        )
    elif mode == "column":
        out = weights / ivy.maximum(
            ivy.vector_norm(weights, axis=-2, keepdims=True, ord=ord), eps
        )
    elif mode == "row_column":
        out = weight_conditioning(weights, mode="row", eps=eps, ord=ord)
        out = weight_conditioning(out, mode="column", eps=eps, ord=ord)
    else:
        raise ValueError(
            f"mode must be one of 'row', 'column', 'row_column', got {mode!r}"
        )
    return out


class WeightConditioning(Module):
    def __init__(
        self,
        /,
        *,
        mode: str = "row",
        eps: float = 1e-12,
        ord=2,
        device=None,
        v=None,
        dtype=None,
    ):
        """Class for conditioning weight matrices by diagonal equilibration.

        Reuses the tensor -> tensor ``_forward`` contract of :class:`LayerNorm`:
        the layer takes a weight matrix and returns its conditioned form, with
        the per-row (or per-column) preconditioner recomputed on every forward
        pass. The layer holds no learnable parameters -- the preconditioner is
        a function of the input weights, mirroring the paper's parameter-free
        formulation.

        Parameters
        ----------
        mode
            ``"row"`` scales each row to unit norm (default, the paper's
            primary form), ``"column"`` scales each column, ``"row_column"``
            applies both.
        eps
            Floor applied to each norm to avoid division by zero.
            Default is ``1e-12``.
        ord
            Order of the vector norm used for equilibration. Default is ``2``.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1',
            'cpu' etc. (Default value = None)
        v
            the variables for each submodule in the sequence, constructed
            internally by default.
        """
        if mode not in ("row", "column", "row_column"):
            raise ValueError(
                f"mode must be one of 'row', 'column', 'row_column', got {mode!r}"
            )
        self._mode = mode
        self._epsilon = eps
        self._ord = ord
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _forward(self, weights):
        """Condition ``weights`` by recomputing the equilibration preconditioner.

        Parameters
        ----------
        weights
            Weight matrix (or stack of matrices) to equilibrate.

        Returns
        -------
        ret
            The conditioned weight matrix, with each row (or column) scaled to
            unit norm.
        """
        return weight_conditioning(
            weights, mode=self._mode, eps=self._epsilon, ord=self._ord
        )

    def _extra_repr(self) -> str:
        return f"mode={self._mode}, epsilon={self._epsilon}, ord={self._ord}"
