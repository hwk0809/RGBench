import torch
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch installation path: {torch.__file__}")

try:
    from torch.utils.cpp_extension import CppExtension
    print("\nSUCCESS: torch.utils.cpp_extension was found successfully!")
except ModuleNotFoundError as e:
    print(f"\nFAILURE: Could not find cpp_extension. Error: {e}")
    print("This confirms the PyTorch installation is broken or incomplete.")