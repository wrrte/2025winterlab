from ultralytics.models import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # 0, 1

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLO('../weights/bd/yolo11m.pt')

model.train(data='yolov11.yaml', epochs=1000, batch=32, imgsz=640)