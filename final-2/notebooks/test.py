import os
import sys
import platform
import subprocess

print("=" * 80)
print("SYSTEM")
print("=" * 80)

print("Python version :", sys.version)
print("Platform       :", platform.platform())
print("Machine        :", platform.machine())
print("Processor      :", platform.processor())

print()

print("=" * 80)
print("PYTORCH")
print("=" * 80)

try:
    import torch

    print("Torch version          :", torch.__version__)
    print("Torch CUDA version     :", torch.version.cuda)
    print("CUDA available         :", torch.cuda.is_available())
    print("cuDNN enabled          :", torch.backends.cudnn.enabled)

    if torch.cuda.is_available():
        print("GPU count              :", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}")
            print("Name                  :", torch.cuda.get_device_name(i))
            print(
                "Memory (GB)           :",
                round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2)
            )

except Exception as e:
    print("Torch import failed:")
    print(e)

print()

print("=" * 80)
print("TORCH-SCATTER")
print("=" * 80)

try:
    import torch_scatter

    print("torch_scatter FOUND")
    print("Version :", getattr(torch_scatter, "__version__", "unknown"))

except Exception as e:
    print("torch_scatter NOT FOUND")
    print("Error:", e)

print()

print("=" * 80)
print("OPEN3D")
print("=" * 80)

try:
    import open3d

    print("open3d FOUND")
    print("Version :", open3d.__version__)

except Exception as e:
    print("open3d NOT FOUND")
    print("Error:", e)

print()

print("=" * 80)
print("NEURALOP")
print("=" * 80)

try:
    import neuralop

    print("neuralop FOUND")
    print("Version :", getattr(neuralop, "__version__", "unknown"))

except Exception as e:
    print("neuralop import failed")
    print("Error:", e)

print()

print("=" * 80)
print("NVIDIA-SMI")
print("=" * 80)

try:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True
    )

    print(result.stdout)

except Exception as e:
    print("Could not run nvidia-smi")
    print(e)

print()

print("=" * 80)
print("TEST CUDA TENSOR")
print("=" * 80)

try:
    import torch

    if torch.cuda.is_available():

        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")

        z = x @ y

        print("CUDA tensor test PASSED")
        print("Tensor device :", z.device)

    else:
        print("CUDA not available")

except Exception as e:
    print("CUDA tensor test FAILED")
    print(e)

print()

print("=" * 80)
print("ENVIRONMENT VARIABLES")
print("=" * 80)

for key in [
    "CUDA_HOME",
    "CUDA_PATH",
    "LD_LIBRARY_PATH",
    "PATH",
]:
    print(f"\n{key}:")
    print(os.environ.get(key, "<not set>"))