import numpy as np
import torch
import pytest
from . import generate_random_test_params
from ne_torch import Linear

test_params = generate_random_test_params(params_list=['in_f', 'out_f', 'batch'])


@pytest.mark.parametrize("in_f,out_f,batch", test_params)
@pytest.mark.parametrize("bias", [True, False])
def test_backward_equivalence(in_f, out_f, batch, bias):
    # Torch layer
    torch_layer = torch.nn.Linear(in_f, out_f, bias=bias)
    torch_layer.weight.data.uniform_(-1, 1)
    if bias:
        torch_layer.bias.data.uniform_(-1, 1)

    # NumPy layer
    np_layer = Linear(in_f, out_f, bias=bias)
    np_layer.W = torch_layer.weight.detach().numpy().copy()
    if bias:
        np_layer.b = torch_layer.bias.detach().numpy().copy()

    # Inputs
    x_np = np.random.randn(batch, in_f).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # Forward
    out_np = np_layer.forward(x_np)
    out_torch = torch_layer(x_torch)

    # Random grad from next layer
    grad_out_np = np.random.randn(batch, out_f).astype(np.float32)
    grad_out_torch = torch.tensor(grad_out_np)

    # Backward
    grad_x_np = np_layer.backward(grad_out_np)
    out_torch.backward(grad_out_torch)

    # Compare weight gradients
    assert np.allclose(np_layer.dW, torch_layer.weight.grad.numpy(), atol=1e-4)

    # Compare bias gradients
    if bias:
        assert np.allclose(np_layer.db, torch_layer.bias.grad.numpy(), atol=1e-4)

    # Compare gradient wrt input
    assert np.allclose(grad_x_np, x_torch.grad.numpy(), atol=1e-4)
