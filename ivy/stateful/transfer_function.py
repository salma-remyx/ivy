"""State-space sequence mixing via a frequency-domain transfer function.

This module implements the *state-free inference* mechanism of a diagonal
state-space model (SSM). Instead of unrolling the recurrence
``s_t = A s_{t-1} + B x_t`` step by step -- whose memory cost grows with the
state size -- the model is described by its transfer function ``H(z)`` in the
frequency domain and applied to the whole sequence at once with a single pair
of FFTs. The compute and memory of the forward pass are then independent of the
state size, which is the central result of the approach.

The transfer function of a diagonal SSM factorises per pole, so the frequency
response at the ``L`` FFT frequency points is

``H_k = D + C @ diag(1 / (1 - p_s z_k)) @ B``

where ``z_k = exp(-2 pi i k / L)`` and ``p_s`` are the (complex) poles. The
output is the inverse FFT of the element-wise product ``H_k * FFT(x)``, sliced
to the original length, which is the causal linear convolution of the input
with the filter's impulse response.

Adapted from "State-Free Inference of State-Space Models: The Transfer Function
Approach" (arXiv:2405.06147). The core mechanism -- a frequency-domain transfer
function parametrisation evaluated with FFTs for state-free parallel inference
-- is preserved at full fidelity; the surrounding training machinery and
benchmark suite of the paper are intentionally out of scope.
"""

# local
import ivy


def _next_power_of_two(n):
    """Return the smallest power of two greater than or equal to ``n``."""
    p = 1
    while p < n:
        p *= 2
    return p


def transfer_function_poles(pole_logit, pole_angle):
    """Parametrise stable complex poles of the transfer function.

    The poles are expressed as ``radius * exp(i angle)`` with
    ``radius = sigmoid(pole_logit)`` in ``(0, 1)``. Bounding the radius keeps
    every pole strictly inside the unit disc, so the impulse response decays
    and the finite-length FFT convolution is well conditioned.

    Parameters
    ----------
    pole_logit
        Pre-activation logit of each pole radius *[state_size]*.
    pole_angle
        Angle (phase) of each pole *[state_size]*.

    Returns
    -------
    ret
        Complex poles *[state_size]* with magnitude in ``(0, 1)``.
    """
    radius = ivy.sigmoid(pole_logit)
    return ivy.multiply(
        radius,
        ivy.exp(ivy.complex(ivy.zeros_like(pole_angle), pole_angle)),
    )


def apply_transfer_function(x, poles, input_mix, output_mix, feedthrough):
    """Apply a diagonal state-space transfer function to a sequence.

    State-free, parallel inference: the whole sequence is processed with a pair
    of FFTs, so the cost does not scale with the state size. The input is
    zero-padded so that the circular FFT convolution equals the causal linear
    convolution, and the first ``seq_len`` samples are returned.

    Parameters
    ----------
    x
        Input sequence *[batch_shape, seq_len, input_channels]*.
    poles
        Complex poles of the transfer function *[state_size]*, each inside the
        unit disc.
    input_mix
        Input-to-state matrix ``B`` *[state_size, input_channels]*.
    output_mix
        State-to-output matrix ``C`` *[output_channels, state_size]*.
    feedthrough
        Direct (feed-through) matrix ``D`` *[output_channels, input_channels]*.

    Returns
    -------
    ret
        Output sequence *[batch_shape, seq_len, output_channels]*.
    """
    seq_len = x.shape[-2]
    fft_len = _next_power_of_two(2 * seq_len)

    # Frequency grid z_k = exp(-2 pi i k / L) matching the forward FFT sign.
    k = ivy.arange(fft_len)
    z = ivy.exp(ivy.complex(ivy.zeros((fft_len,)), -2.0 * ivy.pi * k / fft_len))

    # Per-pole, per-frequency kernel 1 / (1 - p_s z_k) of shape (fft_len, state).
    kernel = 1.0 / (1.0 - poles[None, :] * z[:, None])

    # Matrix-valued frequency response H_k = C diag(kernel_k) B + D.
    b_complex = ivy.complex(input_mix, ivy.zeros_like(input_mix))
    c_complex = ivy.complex(output_mix, ivy.zeros_like(output_mix))
    response = ivy.einsum("os,ks,si->oik", c_complex, kernel, b_complex)
    response = response + feedthrough[..., None]  # (out, in, fft_len)

    # Zero-pad the sequence so the circular convolution is the causal one.
    pad = ivy.zeros(x.shape[:-2] + (fft_len - seq_len,) + x.shape[-1:], dtype=x.dtype)
    spectrum = ivy.fft(ivy.concat([x, pad], axis=-2), -2)

    output_spectrum = ivy.einsum("oik,...ki->...ko", response, spectrum)
    output = ivy.ifft(output_spectrum, -2)
    return ivy.real(output[..., :seq_len, :])
