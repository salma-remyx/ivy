"""Collection of tests for spectral analysis of weight containers."""

# global
import numpy as np

# local
import ivy
from ivy.stateful.weight_spectrum import (
    WeightSpectrumTracker,
    container_spectral_summary,
    spectral_summary,
)


def test_spectral_summary_matches_manual_svd():
    # Diagonal matrix, so the singular values are the sorted |diagonal|.
    w = ivy.array(np.diag(np.array([3.0, 1.0, 0.5, 0.1])).astype(np.float32))
    summary = spectral_summary(w)
    np.testing.assert_allclose(
        ivy.to_numpy(ivy.asarray(summary["s"])), [3.0, 1.0, 0.5, 0.1], rtol=1e-5
    )
    assert float(ivy.to_numpy(summary["sigma_max"])) == 3.0
    np.testing.assert_allclose(
        float(ivy.to_numpy(summary["fro_norm"])),
        float(np.sqrt(np.sum(np.array([3.0, 1.0, 0.5, 0.1]) ** 2))),
        rtol=1e-5,
    )
    # stable rank = ||w||_F^2 / sigma_max^2
    np.testing.assert_allclose(
        float(ivy.to_numpy(summary["stable_rank"])),
        float(np.sum(np.array([3.0, 1.0, 0.5, 0.1]) ** 2) / 9.0),
        rtol=1e-5,
    )
    # tail mass excludes the leading singular value
    np.testing.assert_allclose(
        float(ivy.to_numpy(summary["tail_mass_ratio"])),
        float((1.0 + 0.25 + 0.01) / (9.0 + 1.0 + 0.25 + 0.01)),
        rtol=1e-5,
    )


def test_container_spectral_summary_skips_vector_parameters():
    v = ivy.Container(
        {
            "w": ivy.array(np.eye(4, dtype=np.float32) * 2.0),
            "b": ivy.array(np.zeros(4, dtype=np.float32)),
        }
    )
    summaries = container_spectral_summary(v)
    # only the matrix parameter is summarised, and each summary expands into
    # one sub-container of fields
    assert set(summaries.cont_to_iterator_keys()) == {
        f"w/{field}"
        for field in (
            "s",
            "sigma_max",
            "fro_norm",
            "stable_rank",
            "tail_mass_ratio",
        )
    }
    # Identity scaled by 2: every singular value is 2, so the bulk carries all
    # the energy and the stable rank equals the rank.
    np.testing.assert_allclose(
        float(ivy.to_numpy(summaries.w.sigma_max)), 2.0, rtol=1e-5
    )
    np.testing.assert_allclose(
        float(ivy.to_numpy(summaries.w.stable_rank)), 4.0, rtol=1e-4
    )


def test_optimizer_step_records_spectrum_history():
    # Exercises the track_spectrum hook wired into Optimizer.step.
    rng = np.random.default_rng(0)
    v = ivy.Container(
        {
            "w": ivy.array(rng.standard_normal((6, 4)).astype(np.float32)),
            "b": ivy.array(np.zeros(4, dtype=np.float32)),
        }
    )
    tracker = WeightSpectrumTracker(top_k=2)
    opt = ivy.optimizers.SGD(1e-2, track_spectrum=tracker)
    for _ in range(3):
        grads = ivy.Container(
            {
                "w": ivy.array(rng.standard_normal((6, 4)).astype(np.float32)),
                "b": ivy.array(np.ones(4, dtype=np.float32)),
            }
        )
        v = opt.step(v, grads)

    history = tracker.history
    assert len(history) == 3
    assert [entry["step"] for entry in history] == [0, 1, 2]
    assert all(entry["key_chain"] == "w" for entry in history)
    assert all("s_0" in entry and "s_1" in entry for entry in history)
    for entry in history:
        assert entry["sigma_max"] >= entry["s_0"] - 1e-6
        assert entry["s_0"] >= entry["s_1"]
        assert entry["fro_norm"] > 0

    drift = tracker.spectrum_drift()
    assert set(drift) == {
        "sigma_max",
        "fro_norm",
        "stable_rank",
        "tail_mass_ratio",
    }


def test_tracker_numbers_steps_across_multiple_parameters():
    # With more than one matrix parameter, each optimizer step must contribute
    # one history entry per parameter, all sharing that step's index.
    rng = np.random.default_rng(1)
    v = ivy.Container(
        {
            "layer0": {"w": ivy.array(rng.standard_normal((4, 4)).astype(np.float32))},
            "layer1": {"w": ivy.array(rng.standard_normal((4, 4)).astype(np.float32))},
        }
    )
    tracker = WeightSpectrumTracker()
    opt = ivy.optimizers.SGD(1e-2, track_spectrum=tracker)
    for _ in range(2):
        grads = ivy.Container(
            {
                "layer0": {
                    "w": ivy.array(rng.standard_normal((4, 4)).astype(np.float32))
                },
                "layer1": {
                    "w": ivy.array(rng.standard_normal((4, 4)).astype(np.float32))
                },
            }
        )
        v = opt.step(v, grads)

    assert len(tracker.history) == 4
    assert [entry["step"] for entry in tracker.history] == [0, 0, 1, 1]
    assert len({entry["key_chain"] for entry in tracker.history}) == 2
    assert set(tracker.spectrum_drift()) == {
        "sigma_max",
        "fro_norm",
        "stable_rank",
        "tail_mass_ratio",
    }


def test_optimizer_without_tracker_is_unaffected():
    v = ivy.Container({"w": ivy.array(np.eye(3, dtype=np.float32))})
    grads = ivy.Container({"w": ivy.array(np.eye(3, dtype=np.float32))})
    opt = ivy.optimizers.SGD(1e-2)
    ret = opt.step(v, grads)
    assert ivy.Container(ret) is not None
    # gradient descent moves the weight against the gradient
    assert float(ivy.to_numpy(ret.w[0, 0])) < 1.0


def test_tracker_accepts_module_variables():
    class _Linear(ivy.Module):
        def __init__(self, in_features, out_features):
            self._in_features = in_features
            self._out_features = out_features
            ivy.Module.__init__(self)

        def _create_variables(self, device, dtype):
            return {
                "w": ivy.random_normal(
                    shape=(self._out_features, self._in_features),
                    seed=0,
                    device=device,
                    dtype=dtype,
                )
            }

        def _forward(self, x):
            return x @ ivy.swapaxes(self.v.w, -1, -2)

    module = _Linear(4, 6)
    tracker = WeightSpectrumTracker()
    tracker.record(module.v)
    assert len(tracker.history) == 1
    assert tracker.history[0]["key_chain"] == "w"
    assert tracker.history[0]["s_0"] > 0
