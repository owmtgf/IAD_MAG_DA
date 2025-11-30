import numpy as np


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        # Match PyTorch initialization: Kaiming uniform for weights, bias uniform
        sigma = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.normal(0.0, sigma, (out_features, in_features)).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None

        # Cache for backward
        self.x = None

    def forward(self, x):
        """
        x: (batch, in_features)
        returns: (batch, out_features)
        """
        self.x = x
        out = x @ self.W.T
        if self.b is not None:
            out += self.b
        return out

    def backward(self, grad_out):
        """
        grad_out: gradient from next layer, shape (batch, out_features)
        returns: gradient wrt input x, shape (batch, in_features)
        """
        # Gradient wrt weights
        self.dW = grad_out.T @ self.x
        
        # Gradient wrt bias
        if self.b is not None:
            self.db = grad_out.sum(axis=0)

        # Gradient wrt input
        grad_x = grad_out @ self.W
        return grad_x

    def step(self, lr):
        """Simple SGD update."""
        self.W -= lr * self.dW
        if self.b is not None:
            self.b -= lr * self.db


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


class Softmax:
    def __init__(self):
        self.out = None   # store softmax output for backward

    def forward(self, x):
        """
        x: logits, shape (batch, num_classes)
        returns: probabilities (batch, num_classes)
        """
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_out):
        """
        grad_out: upstream gradient dL/dy, shape (batch, num_classes)
        returns: dL/dx, shape (batch, num_classes)
        """
        # softmax Jacobian vectorization:
        # dL/dx = softmax(x) * (grad_out - sum(grad_out * softmax(x)))

        s = self.out 
        dot = np.sum(grad_out * s, axis=1, keepdims=True)
        grad_x = s * (grad_out - dot)
        return grad_x


class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """
        y_pred: (batch, features)
        y_true: (batch, features)
        """
        self.y_pred = y_pred
        self.y_true = y_true
        logit = ((y_pred - y_true) ** 2).mean()
        return logit

    def backward(self):
        """
        Gradient wrt y_pred, same shape as y_pred.
        """
        grad = (2 / self.y_pred.size) * (self.y_pred - self.y_true)
        return grad
