import numpy as np
import torch
import pytest
from . import generate_random_test_params
from ne_torch import MSELoss

test_params = generate_random_test_params(params_list=['batch', 'features'])

# -------------------------
# MSE Loss Tests
# -------------------------

@pytest.mark.parametrize("batch,features", test_params)
def test_mse_loss_forward_backward(batch, features):
    y_pred_np = np.random.randn(batch, features).astype(np.float32)
    y_true_np = np.random.randn(batch, features).astype(np.float32)

    y_pred_torch = torch.tensor(y_pred_np, requires_grad=True)
    y_true_torch = torch.tensor(y_true_np)

    # Forward
    mse_np = MSELoss()
    loss_np = mse_np.forward(y_pred_np, y_true_np)
    loss_torch = torch.nn.functional.mse_loss(y_pred_torch, y_true_torch, reduction="mean")
    assert np.allclose(loss_np, loss_torch.detach().numpy(), atol=1e-6)

    # Backward
    grad_np = mse_np.backward()
    loss_torch.backward()
    assert np.allclose(grad_np, y_pred_torch.grad.numpy(), atol=1e-6)