import torch
import torch_directml as tdm

if tdm.is_available():
    print("DirectML is available.")
else:
    print("DirectML is not available.")

dml = tdm.device()

tensor1 = torch.tensor([1]).to(dml)  # Note that dml is a variable, not a string!
tensor2 = torch.tensor([2]).to(dml)

dml_algebra = tensor1 + tensor2
dml_algebra.item()

print(dml_algebra.item())
