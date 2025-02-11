import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ultralytics.models import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set GPU number

# Paths
weights_path = "../weights/bd_pretrained/yolo11n.pt"
save_dir = "../weights/bd_trained"
dataset_yaml = "../ultralytics/yolov11.yaml"

# Initialize with pretrained weights
model = YOLO(model=weights_path)

#Train YOLO model
model.train(
    data=dataset_yaml,
    epochs=1000,
    batch=128,
    imgsz=640,
    project=save_dir,
    device="cuda",
    # workers=16,  # Increase workers for better data loading
    # amp=False  # Enable automatic mixed precision (faster training)
)
