from ultralytics.models import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"]="5" # 0, 1

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLO(model='/mnt/hdd_4A/choemj/2025winterlab/weights/bd/yolo11x.pt')

model.train(data='yolov11.yaml', epochs=1000, batch=16, imgsz=640)