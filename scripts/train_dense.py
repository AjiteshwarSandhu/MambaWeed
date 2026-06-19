"""
Dense MambaWeed Training

Injects FourDirectionalMamba blocks after all
C2f stages and performs two-stage training:

1. Mamba warmup
2. Full-network finetuning
"""

import copy
from multiprocessing import freeze_support
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.nn.modules import C2f
from ultralytics.models.yolo.detect import DetectionTrainer

from models.mamba_block import FourDirectionalMamba


# ============================================================
# Configuration
# ============================================================

BASELINE = r"D:\GenAI-WDSS\yolov8m.pt"
DATA = r"D:\GenAI-WDSS\data\real\dataset_v2\data.yaml"
PROJECT = r"D:\GenAI-WDSS\runs\detect\experiments"


# ============================================================
# Dense Mamba Injection
# ============================================================

def inject_mamba_dense(model):
    layers = model.model.model
    new_layers = nn.ModuleList()

    c2f_idx = -1

    for layer in layers:
        new_layers.append(layer)

        if isinstance(layer, C2f):
            c2f_idx += 1

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
# Custom Trainer
# ============================================================

class MambaTrainer(DetectionTrainer):

    def save_model(self):

        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": copy.deepcopy(self.model).half(),
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),
            "date": __import__("datetime").datetime.now().isoformat(),
        }

        torch.save(ckpt, self.last)

        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)

        if (
            self.save_period > 0
            and self.epoch % self.save_period == 0
        ):
            torch.save(
                ckpt,
                self.wdir / f"epoch{self.epoch}.pt"
            )


# ============================================================
# Checkpoint Loader
# ============================================================

def load_mamba_ckpt(path):

    ckpt = torch.load(path, map_location="cpu")
    model = ckpt["model"].float()

    return model, ckpt


# ============================================================
# Main Training
# ============================================================

def main():

    print("Loading baseline YOLOv8m...")
    model = YOLO(BASELINE)

    print("\nInjecting dense Mamba blocks...")
    model = inject_mamba_dense(model)

    # --------------------------------------------------------
    # Phase 1: Mamba Warmup
    # --------------------------------------------------------

    print("\nPhase 1: Mamba Warmup")

    trainable_keys = [
        "row_lr",
        "row_rl",
        "col_tb",
        "col_bt",
        "gamma",
        "bn",
    ]

    for name, param in model.model.named_parameters():
        param.requires_grad = any(
            key in name for key in trainable_keys
        )

    model.train(
        data=DATA,
        epochs=10,
        imgsz=640,
        batch=8,
        optimizer="AdamW",
        lr0=1e-3,
        warmup_epochs=2,
        device=0,
        project=PROJECT,
        name="mambaweed_dense_phase1",
        exist_ok=True,
        trainer=MambaTrainer,
    )

    # --------------------------------------------------------
    # Phase 2: Full Finetuning
    # --------------------------------------------------------

    print("\nPhase 2: Full Finetuning")

    phase1_best = (
        Path(PROJECT)
        / "mambaweed_dense_phase1"
        / "weights"
        / "best.pt"
    )

    nn_model, _ = load_mamba_ckpt(phase1_best)

    model2 = YOLO(BASELINE)
    model2.model = nn_model.to("cuda:0")

    for param in model2.model.parameters():
        param.requires_grad = True

    model2.train(
        data=DATA,
        epochs=100,
        imgsz=640,
        batch=8,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        weight_decay=5e-4,
        cos_lr=True,
        patience=30,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.15,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        cls=1.0,
        device=0,
        project=PROJECT,
        name="mambaweed_dense_phase2",
        exist_ok=True,
        save_period=10,
        trainer=MambaTrainer,
    )

    final_model = (
        Path(PROJECT)
        / "mambaweed_dense_phase2"
        / "weights"
        / "best.pt"
    )

    print(f"\nTraining Complete.")
    print(f"Best model: {final_model}")


if __name__ == "__main__":
    freeze_support()
    main()