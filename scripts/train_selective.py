"""
Selective MambaWeed Training

Injects FourDirectionalMamba blocks after selected
deep C2f stages and performs two-stage finetuning:

1. Mamba warmup
2. Full-network finetuning
"""

from multiprocessing import freeze_support

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules import C2f

from models.mamba_block import FourDirectionalMamba


# ============================================================
# Inject Mamba into selected deep stages
# ============================================================
def inject_selective_mamba(model):
    layers = model.model.model
    new_layers = nn.ModuleList()

    c2f_idx = -1

    # Selective deep-stage insertion
    target_indices = [6, 7]

    for layer in layers:
        new_layers.append(layer)

        if isinstance(layer, C2f):
            c2f_idx += 1

            if c2f_idx in target_indices:

                try:
                    channels = layer.cv2.conv.out_channels
                except AttributeError:
                    channels = layer.cv2.out_channels

                print(
                    f"Injecting FourDirectionalMamba "
                    f"after C2f #{c2f_idx} ({channels} channels)"
                )

                new_layers.append(
                    FourDirectionalMamba(channels)
                )

    model.model.model = new_layers
    return model


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    freeze_support()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    BASELINE = r"D:\GenAI-WDSS\runs\detect\experiments\baseline_v2_14class\weights\best.pt"
    DATA = r"D:\GenAI-WDSS\data\real\dataset_v2\data.yaml"

    PROJECT = r"D:\GenAI-WDSS\runs\detect\experiments"

    # --------------------------------------------------------
    # Load baseline
    # --------------------------------------------------------
    print("Loading baseline model...")
    model = YOLO(BASELINE)

    print("Injecting selective Mamba...")
    model = inject_selective_mamba(model)

    # ========================================================
    # Phase 1: Mamba Warmup
    # ========================================================
    print("\nPhase 1: Warmup")

    for name, param in model.model.named_parameters():

        param.requires_grad = False

        if (
            "row_lr" in name
            or "row_rl" in name
            or "col_tb" in name
            or "col_bt" in name
            or "gamma" in name
            or "bn" in name
        ):
            param.requires_grad = True

    model.train(
        data=DATA,
        epochs=8,
        imgsz=640,
        batch=8,
        optimizer="AdamW",
        lr0=3e-4,
        warmup_epochs=2,
        device=0,
        project=PROJECT,
        name="mambaweed_phase1",
        exist_ok=True,
    )

    # ========================================================
    # Phase 2: Full Finetuning
    # ========================================================
    print("\nPhase 2: Full Finetuning")

    for p in model.model.parameters():
        p.requires_grad = True

    model.train(
        data=DATA,
        epochs=50,
        patience=15,
        imgsz=640,
        batch=8,
        optimizer="AdamW",
        lr0=5e-5,
        cos_lr=True,
        mosaic=0.7,
        mixup=0.10,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.3,
        device=0,
        project=PROJECT,
        name="mambaweed_final",
        exist_ok=True,
        save_period=5,
    )

    print("\nTraining Complete.")

    print(
        r"""
Evaluate with:

yolo detect val ^
model=D:\GenAI-WDSS\runs\detect\experiments\mambaweed_final\weights\best.pt ^
data=D:\GenAI-WDSS\data\real\dataset_v2\data.yaml ^
split=test device=0 verbose=True
"""
    )