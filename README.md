# torchastic
stochastic bfloat16 based optimizer library



## Stochastic Optimizer: Reduced Memory Consumption with BF16 Training

### Key Feature: Drastically Lower Memory Requirements

The **Stochastic Optimizer** is designed to significantly reduce memory consumption by training entirely in **BF16 (bfloat16)** precision. Unlike traditional FP32 (32-bit floating point) or mixed precision training, which can still require large memory resources, the Stochastic Optimizer reduces the memory footprint across parameters, gradients, and optimizer states.

By leveraging BF16 for training, the Stochastic Optimizer reduces memory consumption by **50%**, making it ideal for training larger models or increasing batch sizes within the same memory budget.

### Memory Comparison: Adam, Adam8bit, and Stochastic Adam

#### Traditional Adam (FP32)
- **Parameter storage**: 4 bytes per parameter
- **Gradient storage**: 4 bytes per gradient
- **State 1 (momentum)**: 4 bytes per parameter
- **State 2 (variance)**: 4 bytes per parameter

Total memory required:  
**4 + 4 + 4 + 4 = 16 bytes per parameter**

#### Adam8bit (Mixed Precision)
- **Parameter storage**: 4 bytes per parameter
- **Gradient storage**: 4 bytes per gradient
- **State 1 (momentum)**: 1 byte per parameter (quantized)
- **State 2 (variance)**: 1 byte per parameter (quantized)

Total memory required:  
**4 + 4 + 1 + 1 = 10 bytes per parameter**

#### Stochastic Adam (BF16)
In contrast, **Stochastic Adam** optimizes memory usage by storing everything in **BF16**:
- **Parameter storage**: 2 bytes per parameter
- **Gradient storage**: 2 bytes per gradient
- **State 1 (momentum)**: 2 bytes per parameter
- **State 2 (variance)**: 2 bytes per parameter

Total memory required:  
**2 + 2 + 2 + 2 = 8 bytes per parameter**

### Summary of Memory Savings

- **Traditional Adam**: 16 bytes per parameter
- **Adam8bit**: 10 bytes per parameter (37.5% reduction over Adam)
- **Stochastic Adam**: 8 bytes per parameter (50% reduction over Adam, 20% reduction over Adam8bit)

With **Stochastic Adam**, you save an additional 20% of memory compared to **Adam8bit**, while maintaining the precision needed for training stability and accuracy.

### Why BF16?

BF16 (bfloat16) is advantageous for deep learning because it provides the **same dynamic range as FP32**, while using fewer bits for the mantissa (7 bits in BF16 vs. 23 bits in FP32). This allows models to represent a wide range of values while using half the memory of FP32. However, the reduced precision of the mantissa can sometimes lead to **stale gradients**, especially during long accumulation phases like weight updates, where small updates can become too insignificant to register in BF16.

### Stochastic BF16: Solving Stale Gradients

A key innovation in **Stochastic BF16** is the use of **stochastic rounding** when casting from FP32 to BF16. Stochastic rounding ensures that even very small updates, which might be lost due to BF16's reduced precision, are probabilistically rounded up or down based on the FP32 value. This prevents the common issue of **stale gradients** where updates become too small to affect the model during long training accumulations. ([Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192)) 

Thanks to @Nerogar for fast stochastic rounding pytorch implementation!

#### Benefits of Stochastic Rounding:
- **Prevents stale gradients**: Small updates that could otherwise be lost are preserved, ensuring more accurate weight updates over time. 
- **Improved training stability**: Stochastic rounding is particularly useful during weight updates, where stable accumulation is critical to maintaining convergence and preventing stalling in training.

### Conclusion

The **Stochastic Optimizer** provides a more memory-efficient alternative to both **Adam** and **Adam8bit** by training entirely in BF16 precision. It reduces the memory footprint by **50%** compared to FP32 Adam, and by **20%** compared to Adam8bit, while also preventing stale gradients through stochastic rounding. This makes the Stochastic Optimizer an excellent choice for scaling up models and improving efficiency in resource-constrained environments, all without sacrificing the quality of your training process.



## How to Install
`pip install torchastic`

## Build from Source
```
git clone https://github.com/lodestone-rock/torchastic/
cd torchastic
python setup.py sdist bdist_wheel
pip install .
```

## How to Use
```py
import torch
import torch.nn as nn
from torchastic import AdamW, StochasticAccumulator


class Model(nn.Module):
    ...


# Init model
model = Model(*model_args)
model.to(torch.bfloat16)
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)

# Apply stochastic grad accumulator hooks
StochasticAccumulator.assign_hooks(model)

# Training
while True:

    # Gradient accumulation
    for _ in range(grad_accum_length):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.loss(*model_input)
        loss.backward()

    # Apply grad buffer back
    StochasticAccumulator.reassign_grad_buffer(model)
    optimizer.step()
    optimizer.zero_grad()
```
