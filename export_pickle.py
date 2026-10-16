"""
export_pickle.py
================
Exports the trained UNet model to pickle (.pkl) format:
1. model/best_model.pkl (and model/unet_model.pkl)
Allows instant loading without needing to retrain the model again.
Supports both standard Python pickle.load() and torch.load().
"""

import sys
import pickle
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
PTH_PATH = MODEL_DIR / "best_model.pth"
PKL_PATH = MODEL_DIR / "best_model.pkl"
UNET_PKL_PATH = MODEL_DIR / "unet_model.pkl"

from model.unet import UNet

def export_to_pickle():
    if not PTH_PATH.exists():
        print(f"[ERROR] Source model checkpoint {PTH_PATH} not found!")
        sys.exit(1)

    print(f"[INFO] Loading trained weights from {PTH_PATH}...")
    device = torch.device("cpu")
    checkpoint = torch.load(str(PTH_PATH), map_location=device)

    base_features = 8
    val_dice = 0.0
    val_iou = 0.0
    epoch = 0

    if isinstance(checkpoint, dict):
        base_features = checkpoint.get("base_features", 8)
        val_dice = checkpoint.get("val_dice", 0.0)
        val_iou = checkpoint.get("val_iou", 0.0)
        epoch = checkpoint.get("epoch", 0)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    # Instantiate model and load trained state
    model = UNet(in_channels=1, out_channels=1, base_features=base_features)
    model.load_state_dict(state_dict)
    model.eval()

    # Package comprehensive bundle containing:
    # 1. The complete model instance
    # 2. State dictionary
    # 3. Model metadata (base_features, val_dice, val_iou, epoch)
    bundle = {
        "model": model,
        "model_state_dict": model.state_dict(),
        "base_features": base_features,
        "val_dice": val_dice,
        "val_iou": val_iou,
        "epoch": epoch,
        "framework": "PyTorch",
        "format": "pickle"
    }

    # Save bundle to best_model.pkl using Python's standard pickle module
    with open(PKL_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SUCCESS] Saved pickled model bundle to {PKL_PATH} ({PKL_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # Also save direct model instance to unet_model.pkl for standard pickle.load() usage
    with open(UNET_PKL_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SUCCESS] Saved direct pickled model object to {UNET_PKL_PATH} ({UNET_PKL_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # Test loading both with pickle to ensure 100% integrity
    print("[INFO] Verifying pickle files...")
    with open(PKL_PATH, "rb") as f:
        loaded_bundle = pickle.load(f)
        assert "model" in loaded_bundle
        assert isinstance(loaded_bundle["model"], UNet)
    print(f" -> Verified {PKL_PATH.name} loads cleanly with pickle.load (Val Dice: {loaded_bundle['val_dice']:.4f})")

    with open(UNET_PKL_PATH, "rb") as f:
        loaded_model = pickle.load(f)
        assert isinstance(loaded_model, UNet)
    print(f" -> Verified {UNET_PKL_PATH.name} loads cleanly with pickle.load")

    print("\n[ALL DONE] Model successfully serialized to pickle format. Retraining is no longer needed!")

if __name__ == "__main__":
    export_to_pickle()
