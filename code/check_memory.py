import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == 'cuda':
   print("Memory Allocated:", round(torch.cuda.memory_allocated(0)/1024**3, 1), "GB")
   print("Memory Reserved:", round(torch.cuda.memory_reserved(0)/1024**3, 1), "GB")