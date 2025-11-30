import numpy as np
import torch
import pytest
from . import generate_random_test_params
from ne_torch import Linear

test_params = generate_random_test_params(params_list=['in_f', 'out_f', 'batch'])


@pytest.mark.parametrize("in_f,out_f,batch", test_params)
def test_forward_equivalence(in_f, out_f, batch):
    # torch layer
    torch_layer = torch.nn.Linear(in_f, out_f, bias=True)

    # numpy layer
    np_layer = Linear(in_f, out_f)

    # copy parameters from torch → numpy
    np_layer.W = torch_layer.weight.detach().numpy().copy()
    np_layer.b = torch_layer.bias.detach().numpy().copy()

    x_np = np.random.randn(batch, in_f).astype(np.float32)
    x_torch = torch.tensor(x_np)

    out_np = np_layer.forward(x_np)
    out_torch = torch_layer(x_torch).detach().numpy()

    assert np.allclose(out_np, out_torch, atol=1e-5)
