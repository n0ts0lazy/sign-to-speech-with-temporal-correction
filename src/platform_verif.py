import torch
import time

print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))

# Allocate tensors
x = torch.randn(10000, 10000, device="cuda")
y = torch.randn(10000, 10000, device="cuda")

torch.cuda.synchronize()
start = time.time()
z = x @ y
torch.cuda.synchronize()
print("Matrix multiply done in", time.time() - start, "seconds")

