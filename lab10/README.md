# Lab Exercise 10: Learning the XOR Boolean Function Using an MLP

This repository directory contains the implementation of a Multi-Layer Perceptron (MLP) specifically designed to solve the non-linear XOR Boolean function.

## Deliverables
- **`2547218_lab10.ipynb`**: Complete Jupyter Notebook executing the steps, rendering the training curves, predicting outputs, and visualizing the decision boundaries.
- **`2547218_lab10.py`**: Standalone Python implementation with `matplotlib` logic matching the notebook outputs.
- **`2547218_lab10.html`**: Exported HTML summary of the notebook execution.

## Implementation Details
The XOR boolean function is a classic problem that linear classifiers (like a single perceptron) fail to solve. This lab utilizes a hidden layer to map the problem into a linearly separable space.

### 1. Keras (TensorFlow High-Level API)
- **Architecture**:
  - **Input Layer**: 2 neurons (for $X_1, X_2$)
  - **Hidden Layer**: 8 neurons, `ReLU` activation
  - **Output Layer**: 1 neuron, `Sigmoid` activation
- **Compilation**: Adam optimizer (`lr=0.05`), Binary Cross-Entropy loss.
- **Execution**: `model.fit()` abstracted training loops perfectly achieving 100% accuracy within 200 epochs.

### 2. TensorFlow Low-Level API
- Manual implementation defining `W1, b1, W2, b2` using `tf.Variable`.
- Custom forward pass utilizing `tf.matmul` and explicit activation functions (`tf.nn.relu`, `tf.nn.sigmoid`).
- Manual Loss: Computed Binary Cross-Entropy.
- Optimization Loop: Used `tf.GradientTape` to trace backpropagation gradients and applied them via an Adam optimizer sequentially.

## Analysis & Hyperparameters
- **Activation Functions**: The ReLU or Tanh function in the hidden layer provides the non-linearity. Without it, the network behaves strictly as a linear classifier and fails XOR.
- **Hidden Neurons**: Two hidden neurons can theoretically solve XOR by drawing two boundaries. Expanding to 4 or 8 smooths the gradient landscape leading to faster and more reliable convergence.
- **Learning Rate**: A small learning rate (0.001) required upwards of 2,000 epochs, while increasing to 0.05 effectively solved it in around 200 epochs.
- **Visual Analysis**: Decision boundary plots distinctively show non-linear separation curving neatly around the diagonal (0,1 and 1,0 cases versus 0,0 and 1,1 cases).
