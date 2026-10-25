
# Creating a Deep Learning Framework from Scratch with Numpy and Triton Kernels

## Introduction

In modern machine learning, creating models that do certain tasks for you often boils down to using some neural network (which more formally is a composition of linear and non-linear functions), and "training" it with a large dataset of input and output pairs. Training a neural network usually involves computing the gradients of the loss function with respect to the model parameters, and then updating those parameters based on these gradients using some optimization algorithm (like stochastic gradient descent).

Since we can represent a neural network as a composition of functions (usually a series of matrix multiplications and element-wise non-linear operations), we can use the chain rule to compute the gradients of the loss function with respect to the model parameters. This process is known as backpropagation. However, if we were to derive equations for the gradients every time we wanted to implement a new model, it would be very tedious and annoying. Therefore, it would be nice to have a framework that makes this process on computing gradients easier and more automatic.

## How do we design a deep learning framework?

Consider a simple neural network with one hidden layer. We can write out this network as a composition of functions. First, a "hidden layer" is a linear transformation of the input, followed by a non-linear activation function. Then, the output layer is another linear transformation of the hidden layer's output. We can write this as:

$$
\text{hidden}(x) = \text{ReLU}(W_1 x + b_1)
$$

$$
\text{output}(x) = W_2 x + b_2
$$

Thus:
$$
f_\theta(x) = \text{output}(\text{hidden}(x)) = W_2 \text{ReLU}(W_1 x + b_1) + b_2
$$

Here, we can see that the model parameters are $\theta = \{W_1, b_1, W_2, b_2\}$. The loss function is a function of the model's output and the true labels. For example, if we are doing regression, we might use mean squared error (assuming that the true labels are $y$, and the output of the model is a scalar):

$$
\text{loss}(x, y) = \frac{1}{2} (f_\theta(x) - y)^2
$$

Let's step through the process of computing the gradients of the loss function with respect to the model parameters, so for every number in every weight matrix and bias vector, we can compute the partial derivative of the loss function with respect to that number.

So we want:
$$
\frac{\partial \text{loss}}{\partial W_1}, \frac{\partial \text{loss}}{\partial b_1}, \frac{\partial \text{loss}}{\partial W_2}, \frac{\partial \text{loss}}{\partial b_2}