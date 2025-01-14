from ultralytics import YOLO
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 0, 1

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLO('../weights/bd_yolov10x.pt')

model.train(data='yolov10.yaml', epochs=500, batch=64, imgsz=640)