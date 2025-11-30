# Source Code
Code of all required components located in `./ne_torch.py`

# Test Core
To test results similarity of current and PyTorch implementations open terminal then run

```sh
cd ./tests
pytest -v
```

# Training Pipeline
`MNIST` preparation, model assembly and training pipeline in `./train.ipynb`

# Required Fully-conected Neural Network Components Description

Let's took a brief look on implemented modules logic:

## **Linear**

### Description
A fully connected layer performing an affine transformation:

$y = x W^T + b$

- `in_features`: number of input features  
- `out_features`: number of output features  
- `W`: weight matrix of shape `(out_features, in_features)`  
- `b`: bias vector of shape `(out_features,)`

To initialize weights I used `Normal Xavier Initialization`:

$W \sim N(0, \sigma^2)$

where the $\sigma$ is $\sigma = \sqrt{ \frac {2} {n_{input} + n_{output}} }$

### **Forward**
Simply apply weights and biases to input activations $x$:

$y = x W^T + b$

### **Backward**
We should propagate gradients using gathered `y` activation after using `forward`, so we should calculate $\frac {\partial L} {\partial W}$ and $\frac {\partial L} {\partial b}$ gradients for weights and biases of current layer and also $\frac {\partial L} {\partial x}$ for gradient propagation through entire network. I'll drop the batch indexes in following formulas and explain using `batch=1`.

1. Layer weights gradients: 

    $\frac {\partial L} {\partial W} = \frac {\partial L} {\partial y} \frac {\partial y} {\partial W}$, where the $\frac {\partial L} {\partial y}$ given from the loss function backpropagation, we need to found $\frac {\partial y} {\partial W}$

    For certain $y_i$ we got: \
    $\frac {\partial y_j} {\partial W_{ji}} =$ 
    $\frac {\partial (\sum_{i=1}^{\text{in\_features}} W_{ij} x_i + b_j)} {\partial W_{ji}} =$ 
    $\frac {\partial (W_{j1}x_1 + \dots + W_{ij}x_i + \dots)} {\partial W_{ji}} =$ 
    $\frac {\partial W_{ji}} {\partial W_{ji}}x_i = x_i$

    So for all activations $y$ this derivative is equal to $\frac {\partial y} {\partial W} = x$, so the gradient is:
    $$\boxed{\frac {\partial L} {\partial W} = \frac {\partial L} {\partial y} x}$$

2. Bias gradient

    $$\boxed{ \frac {\partial L} {\partial b} = \sum_{j=1}^{\text{out\_features}} \frac {\partial L} {\partial y_j} }$$

3. Activation gradient

    $\frac {\partial L} {\partial x} = \frac {\partial L} {\partial y} \frac {\partial y} {\partial x}$, we need to found $\frac {\partial y} {\partial x}$

    Using the similar pipeline as for weights gradient we got: \
    $\frac {\partial L} {\partial x} = \frac {\partial L} {\partial y} \frac {\partial y} {\partial x}$ \
    and \
    $\frac {\partial y} {\partial x} = W_{ji} \frac {\partial x_i} {\partial x_i} = W_{ji}$, so our gradient is:

    $$\boxed{\frac {\partial L} {\partial x} = \frac {\partial L} {\partial y} W}$$

### **Step**
Gradients application for parameters update:

Simple `Stochastic Gradient Descent` for weights and biases update where we increase/decrese parameters accordingly to collected gradients with a `learning_rate` parameter as gradient multiplicator, responsible for an optimization step size.

---

## ReLU

### **Description**
Rectified Linear Unit activation that zeroes out all negative inputs:

### **Forward**

$y = \text{ReLU}(x) = \max(0, x)$

### **Backward**

$\frac {\partial L} {\partial x} = \frac {\partial L} {\partial y} \frac {\partial y} {\partial x}$ where $\frac {\partial y} {\partial x}$:

$\text{mask} = \frac {\partial y} {\partial x} = 
\begin{cases}
1, x \gt 0 \\
0, x \le 0
\end{cases}$

Gradient:

$$\boxed{\frac {\partial L} {\partial x} = \frac {\partial L} {\partial y} \times \text{mask} }$$


---

## Softmax

### **Description**
The softmax function converts raw logits into normalized probabilities:

$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$

It is typically applied to the final layer of a classifier.

---

### **Forward**

Given logits matrix $X$ of shape `(batch, classes)`:

1. Subtract maximum value for numerical stability  
$
x_{\text{shifted}} = X - \max(X, \text{axis}=1)
$

2. Exponentiate  
$
e^{x_i} = e^{x_{\text{shifted}, i}}
$

3. Normalize  
$
s_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$

### **Backward**
To propagate gradients through softmax, we need:

$
\frac{\partial s_i}{\partial x_j}
$

The Jacobian of softmax is:

$
\frac{\partial s_i}{\partial x_j} =
\begin{cases}
s_i (1 - s_i) & i=j \\ - s_i s_j & i \ne j
\end{cases}
$

Gradient is: \
$
\frac{\partial L}{\partial x_i}
= \sum_j g_j \frac{\partial s_j}{\partial x_i}
$

But note we don't need backward propgation anyways for a simple network implementation, bacause softmax would be used only while logits evaluation. 

---

## MSELoss

### **Description**
Mean Squared Error loss:
$ L = \frac{1}{N} \sum_{i=1}^{N} (y_i - t_i)^2 $

Used for regression or simple numeric targets.

### **Forward**
Stores predictions and targets:

$ L = \frac{1}{N} \sum (\text{pred} - \text{target})^2 $

### **Backward**
Analytical gradient:

$ \frac{\partial L}{\partial y} = \frac{2}{N} (\text{pred} - \text{target}) $ \
Where $N$ is the total number of elements.

Note that this is the exact $\frac{\partial L}{\partial y}$ I mentioned in previous modules.

---

## Summary

These components allow building and training small neural networks without using PyTorch's autograd.  
They demonstrate how forward passes and analytical derivatives form the basis of backpropagation.

