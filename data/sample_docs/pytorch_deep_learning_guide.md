# PyTorch Deep Learning Architecture & Training Loop

## 1. Tensors and Compute Graphs
PyTorch uses `torch.Tensor` as its primary multidimensional data structure, enabling hardware acceleration on CUDA GPUs, Apple Silicon MPS, and CPUs. Tensors maintain computational history dynamically when `requires_grad=True`.

## 2. Automatic Differentiation (Autograd)
PyTorch implements reverse-mode automatic differentiation via `torch.autograd`. When a scalar loss value calls `loss.backward()`, autograd applies the vector-Jacobian product along the directed acyclic graph (DAG) using the calculus chain rule, computing gradients for all leaf tensors.

## 3. Neural Network Modules (torch.nn.Module)
Custom architectures subclass `nn.Module`. Layer components such as `nn.Linear`, `nn.Conv2d`, `nn.LayerNorm`, and `nn.Dropout` are registered in `__init__`. The `forward(self, x)` method specifies the computation pass.

## 4. Standard PyTorch Training Loop
A standard supervised training iteration consists of five deterministic steps:
1. `optimizer.zero_grad()`: Clears accumulated gradients from previous batches.
2. `outputs = model(inputs)`: Executes the forward pass to produce predictions.
3. `loss = criterion(outputs, targets)`: Computes the scalar objective error.
4. `loss.backward()`: Computes parameter gradients via backpropagation.
5. `optimizer.step()`: Updates model weights using the chosen optimization algorithm (e.g. AdamW, SGD).
