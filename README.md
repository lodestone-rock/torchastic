# torchastic
stochastic bfloat16 based optimizer library
## How to Use
```py
import torch
import torch.nn as nn
from torchastic import Compass, StochasticAccumulator


class Model(nn.Module):
    ...


# Init model
model = Model(*model_args)
optimizer = Compass(model.parameters(), lr=0.01, weight_decay=1e-2, amp_fac=5)

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