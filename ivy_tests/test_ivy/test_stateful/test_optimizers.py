"""Collection of tests for Ivy optimizers."""

# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method
from ivy_tests.test_ivy.test_functional.test_core.test_gradients import (
    get_gradient_arguments_with_lr,
)


# adam
@handle_method(
    method_tree="Adam._step",
    dtype_x_lr=get_gradient_arguments_with_lr(
        min_value=1e-05,
        max_value=1e08,
        num_arrays=2,
        float_lr=True,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    beta1_n_beta2_n_epsilon=helpers.list_of_size(
        x=helpers.floats(min_value=1e-1, max_value=1),
        size=3,
    ),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    test_gradients=st.just(True),
)
def test_adam_optimizer(
    dtype_x_lr,
    beta1_n_beta2_n_epsilon,
    inplace,
    stop_gradients,
    on_device,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    test_gradients,
    init_flags,
    method_flags,
):
    input_dtype, x, lr = dtype_x_lr
    beta1, beta2, epsilon = beta1_n_beta2_n_epsilon
    xs_grad_idxs = [[0, 0]] if method_flags.num_positional_args else [[1, "v"]]
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-1,
        atol_=1e-1,
        test_gradients=test_gradients,
        xs_grad_idxs=xs_grad_idxs,
        on_device=on_device,
    )


# AdamW
@handle_method(
    method_tree="AdamW._step",
    dtype_x_lr=get_gradient_arguments_with_lr(
        min_value=1e-05,
        max_value=1e08,
        num_arrays=2,
        float_lr=True,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
    ),
    beta1_n_beta2_n_epsilon=helpers.list_of_size(
        x=helpers.floats(min_value=1e-1, max_value=1),
        size=3,
    ),
    weight_decay=helpers.floats(min_value=0, max_value=1e-1),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    test_gradients=st.just(True),
)
def test_adamw_optimizer(
    dtype_x_lr,
    beta1_n_beta2_n_epsilon,
    weight_decay,
    inplace,
    stop_gradients,
    on_device,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    test_gradients,
    init_flags,
    method_flags,
):
    input_dtype, x, lr = dtype_x_lr
    beta1, beta2, epsilon = beta1_n_beta2_n_epsilon
    xs_grad_idxs = [[0, 0]] if method_flags.num_positional_args else [[1, "v"]]
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "weight_decay": weight_decay,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-1,
        atol_=1e-1,
        test_gradients=test_gradients,
        xs_grad_idxs=xs_grad_idxs,
        on_device=on_device,
    )


# lamb
@handle_method(
    method_tree="LAMB._step",
    dtype_x_lr=get_gradient_arguments_with_lr(
        min_value=-1e5,
        max_value=1e5,
        num_arrays=2,
        float_lr=True,
    ),
    beta1_n_beta2_n_epsilon_n_lambda=helpers.list_of_size(
        x=helpers.floats(
            min_value=1e-2,
            max_value=1.0,
        ),
        size=4,
    ),
    mtr=st.one_of(
        helpers.ints(min_value=1, max_value=10),
        st.floats(min_value=1e-2, max_value=10, exclude_min=True),
    ),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    test_gradients=st.just(True),
)
def test_lamb_optimizer(
    dtype_x_lr,
    beta1_n_beta2_n_epsilon_n_lambda,
    mtr,
    inplace,
    stop_gradients,
    on_device,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    test_gradients,
):
    input_dtype, x, lr = dtype_x_lr
    beta1, beta2, epsilon, decay_lambda = beta1_n_beta2_n_epsilon_n_lambda
    xs_grad_idxs = [[0, 0]] if method_flags.num_positional_args else [[1, "v"]]
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "max_trust_ratio": mtr,
            "decay_lambda": decay_lambda,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-1,
        atol_=1e-1,
        test_gradients=test_gradients,
        xs_grad_idxs=xs_grad_idxs,
        on_device=on_device,
    )


# lars
@handle_method(
    method_tree="LARS._step",
    dtype_x_lr=get_gradient_arguments_with_lr(num_arrays=2, float_lr=True),
    inplace=st.booleans(),
    decay_lambda=helpers.floats(min_value=1e-2, max_value=1.0),
    stop_gradients=st.booleans(),
    test_gradients=st.just(True),
)
def test_lars_optimizer(
    dtype_x_lr,
    decay_lambda,
    inplace,
    stop_gradients,
    on_device,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    test_gradients,
):
    input_dtype, x, lr = dtype_x_lr
    if "bfloat16" in input_dtype:
        test_gradients = False
    xs_grad_idxs = [[0, 0]] if method_flags.num_positional_args else [[1, "v"]]
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "lr": lr,
            "decay_lambda": decay_lambda,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-1,
        atol_=1e-1,
        test_gradients=test_gradients,
        xs_grad_idxs=xs_grad_idxs,
        on_device=on_device,
    )


# sgd
@handle_method(
    method_tree="SGD._step",
    dtype_x_lr=get_gradient_arguments_with_lr(
        min_value=-1e5,
        max_value=1e5,
        num_arrays=2,
        float_lr=True,
    ),
    inplace=st.booleans(),
    stop_gradients=st.booleans(),
    test_gradients=st.just(True),
)
def test_sgd_optimizer(
    dtype_x_lr,
    inplace,
    stop_gradients,
    on_device,
    class_name,
    method_name,
    backend_fw,
    ground_truth_backend,
    init_flags,
    method_flags,
    test_gradients,
):
    input_dtype, x, lr = dtype_x_lr
    xs_grad_idxs = [[0, 0]] if method_flags.num_positional_args else [[1, "v"]]
    helpers.test_method(
        backend_to_test=backend_fw,
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_all_as_kwargs_np={
            "lr": lr,
            "inplace": inplace,
            "stop_gradients": stop_gradients,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "v": x[0],
            "grads": x[1],
        },
        class_name=class_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        xs_grad_idxs=xs_grad_idxs,
        on_device=on_device,
    )


# AdamZ
def test_adamz_optimizer_adapts_learning_rate():
    """Integration: AdamZ is exported from ivy.stateful.optimizers (a non-new
    module) and driven through the existing Optimizer.step machinery, adapting
    the learning rate on overshoot/stagnation (adapted from AdamZ,
    arXiv:2411.15375)."""
    import ivy
    from ivy.stateful.optimizers import AdamZ

    ivy.set_backend("numpy")
    try:
        v = ivy.Container({"w": ivy.array([1.0, -1.0, 2.0])})

        # Overshoot: reversing the gradient direction across two steps makes
        # successive gradients oppose, so the learning rate is reduced.
        opt = AdamZ(lr=0.01)
        opt.step(v, ivy.Container({"w": ivy.array([1.0, 1.0, 1.0])}))
        lr_after_first = opt._lr
        opt.step(v, ivy.Container({"w": ivy.array([-1.0, -1.0, -1.0])}))
        assert opt._lr < lr_after_first

        # Stagnation: tiny gradients for `patience` steps grow the learning rate.
        opt_stag = AdamZ(lr=0.01, stagnation_threshold=1e-3, patience=3)
        tiny = ivy.Container({"w": ivy.array([1e-5, 1e-5, 1e-5])})
        for _ in range(4):
            opt_stag.step(v, tiny)
        assert opt_stag._lr > 0.01

        # The public step returns a valid container with matching structure.
        out = opt.step(v, ivy.Container({"w": ivy.array([0.5, 0.5, 0.5])}))
        assert list(out.keys()) == list(v.keys())
    finally:
        ivy.previous_backend()
