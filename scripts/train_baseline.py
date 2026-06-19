# train_baseline.py
# Clean YOLOv8m baseline — no Mamba injection


from ultralytics import YOLO
from multiprocessing import freeze_support

BASELINE = r"D:\GenAI-WDSS\yolov8m.pt"
DATA     = r"D:\GenAI-WDSS\data\real\dataset_v2\data.yaml"
PROJECT  = r"D:\GenAI-WDSS\runs\detect\experiments"


def main():
    model = YOLO(BASELINE)
    model.info()

    model.train(
        data         = DATA,
        epochs       = 100,
        imgsz        = 640,
        batch        = 8,
        optimizer    = "AdamW",
        lr0          = 1e-4,
        lrf          = 0.01,
        weight_decay = 5e-4,
        cos_lr       = True,
        patience     = 30,
        mosaic       = 1.0,
        close_mosaic = 10,
        mixup        = 0.15,
        hsv_h        = 0.02,
        hsv_s        = 0.7,
        hsv_v        = 0.4,
        translate    = 0.1,
        scale        = 0.5,
        fliplr       = 0.5,
        cls          = 1.0,
        device       = 0,
        project      = PROJECT,
        name         = "yolov8m_baseline",
        exist_ok     = True,
        save_period  = 10,
    )


if __name__ == "__main__":
    freeze_support()
    main()
