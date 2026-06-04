from pathlib import Path
import torch

CHECKPOINT_PATH = Path(
    "/home/dysco/Neeraj/Flow-reconstruction-in-VPM-using-FNO/"
    "final-2/output/task1_v3_simple_training/best_task1_v3_model.pt"
)


def safe_torch_load(path):
    # PyTorch 2.6 changed torch.load's default weights_only behavior.
    # This checkpoint is generated locally by our own notebook, so weights_only=False is acceptable here.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def summarize_value(name, value):
    print(f"\nKey: {name}")
    print("Type:", type(value))

    if torch.is_tensor(value):
        print("Shape:", tuple(value.shape))
        print("Dtype:", value.dtype)
        print("Sample values:", value.flatten()[:10])
        return

    if isinstance(value, dict):
        print("Dict keys:", list(value.keys())[:20])
        if name == "model_state_dict":
            print("State tensors:", len(value))
            for tensor_name, tensor_value in list(value.items())[:12]:
                if torch.is_tensor(tensor_value):
                    print(f"  {tensor_name}: shape={tuple(tensor_value.shape)} dtype={tensor_value.dtype}")
                else:
                    print(f"  {tensor_name}: {type(tensor_value)}")
        else:
            print("Value:", value)
        return

    if isinstance(value, (list, tuple)):
        print("Length:", len(value))
        print("Value:", value)
        return

    print("Value:", value)


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    data = safe_torch_load(CHECKPOINT_PATH)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Top-level type:", type(data))

    if not isinstance(data, dict):
        print("Checkpoint is not a dictionary; raw value:", data)
        return

    print("Keys:", list(data.keys()))
    print("\nImportant metadata")
    for key in ["feature_names", "target_names", "model_config", "settings", "loss_name", "best_epoch", "best_checkpoint_metric"]:
        if key in data:
            summarize_value(key, data[key])

    if "model_state_dict" in data:
        summarize_value("model_state_dict", data["model_state_dict"])


if __name__ == "__main__":
    main()
