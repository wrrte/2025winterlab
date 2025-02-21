from ultralytics.models import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3" # 0, 1

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLO(model='/mnt/hdd_4A/choemj/2025winterlab/type_30000/weight/train/weights/best.pt')

model.train(data='yolov11.yaml', epochs=1, batch=1, imgsz=640, project="11to8")