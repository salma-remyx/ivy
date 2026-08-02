"""Collection of tests for the weight-clipping optimizer."""

# global
import pytest

# local
import ivy
from ivy.stateful import optimizers  # non-new module: the call-site surface
from ivy.stateful import WeightClip, clip_weights


@pytest.fixture(autouse=True)
def numpy_backend():
    """Run on the numpy backend so the test has no heavy framework dependency."""
    ivy.set_backend("numpy")
    yield
    ivy.previous_backend()


def test_weight_clip_projects_after_step():
    # A gradient step that would push weights far outside the bound is pulled
    # back into [-clip_value, clip_value] by the post-update projection.
    opt = WeightClip(optimizers.SGD(lr=1.0), clip_value=0.5)
    v = ivy.Container({"w": ivy.array([2.0, -3.0, 0.1])})
    grads = ivy.Container({"w": ivy.array([1.0, 1.0, 0.0])})

    out = opt.step(v, grads)

    assert ivy.all(ivy.abs(out.w) <= 0.5 + 1e-6).item() is True
    # Values already inside the range are untouched by the clip.
    assert float(out.w[2]) == pytest.approx(0.1, abs=1e-6)


def test_weight_clip_layers_on_top_of_base_optimizer():
    # The base optimizer still advances its own step counter, proving the clip
    # layers on top of the existing learning system rather than bypassing it.
    base = optimizers.Adam(lr=0.1)
    opt = WeightClip(base, clip_value=0.5)
    v = ivy.Container({"w": ivy.array([10.0, -10.0])})
    grads = ivy.Container({"w": ivy.array([1.0, 1.0])})

    for _ in range(3):
        v = opt.step(v, grads)

    assert int(base._count[0]) == 3
    assert ivy.all(ivy.abs(v.w) <= 0.5 + 1e-6).item() is True
    # Without clipping the first weight would be ~9.9; the projection prevents it.
    assert abs(float(v.w[0])) < 1.0


def test_clip_weights_helper_projects_container():
    v = ivy.Container({"w": ivy.array([2.0, -3.0, 0.1])})
    clipped = clip_weights(v, 0.5)
    assert ivy.all(ivy.abs(clipped.w) <= 0.5 + 1e-6).item() is True
