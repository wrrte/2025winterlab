from ultralytics.models import YOLO
import os

# Set GPU number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Paths
weights_path = "weights/bd_pretrained/yolom11.pt"
save_dir = "weights/bd_trained"
dataset_yaml = "ultralytics/yolov11.yaml"

# Initialize with pretrained weights
model = YOLO(model=weights_path)

#Train YOLO model
model.train(
    data=dataset_yaml,
    epochs=1000,
    batch=128,
    imgsz=640,
    project=save_dir  # Save trained model in bd_trained folder
)
