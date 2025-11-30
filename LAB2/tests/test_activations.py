import numpy as np
import torch
import pytest
from . import generate_random_test_params
from ne_torch import ReLU, Softmax

test_params = generate_random_test_params(params_list=['batch', 'features'])

# -------------------------
# ReLU Tests
# -------------------------

@pytest.mark.parametrize("batch,features", test_params)
def test_relu_forward_backward(batch, features):
    relu_np = ReLU()

    x_np = np.random.randn(batch, features).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # Forward
    out_np = relu_np.forward(x_np)
    out_torch = torch.nn.functional.relu(x_torch)
    assert np.allclose(out_np, out_torch.detach().numpy(), atol=1e-6)

    # Backward
    grad_out_np = np.random.randn(batch, features).astype(np.float32)
    grad_out_torch = torch.tensor(grad_out_np)
    
    grad_x_np = relu_np.backward(grad_out_np)
    out_torch.backward(grad_out_torch)

    assert np.allclose(grad_x_np, x_torch.grad.numpy(), atol=1e-6)

# -------------------------
# Softmax Tests
# -------------------------

@pytest.mark.parametrize("batch,features", test_params)
def test_softmax_forward_backward(batch, features):
    # Random input
    x_np = np.random.randn(batch, features).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # -------------------
    # Forward
    # -------------------
    sm_np = Softmax()
    out_np = sm_np.forward(x_np)

    out_torch = torch.nn.functional.softmax(x_torch, dim=1)

    # Compare forward
    assert np.allclose(out_np, out_torch.detach().numpy(), atol=1e-6)

    # -------------------
    # Backward
    # -------------------
    grad_out_np = np.random.randn(batch, features).astype(np.float32)
    grad_out_torch = torch.tensor(grad_out_np, dtype=torch.float32)

    grad_x_np = sm_np.backward(grad_out_np)
    out_torch.backward(grad_out_torch)

    # Compare shapes (full numerical match is tricky without cross-entropy)
    assert grad_x_np.shape == x_torch.grad.numpy().shape

    # Optional: loose check that gradients are roughly similar
    # This is not exact due to PyTorch using autograd precision and batch effects
    np.testing.assert_allclose(grad_x_np, x_torch.grad.numpy(), rtol=1e-4, atol=1e-5)