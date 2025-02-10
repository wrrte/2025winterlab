from ultralytics.models import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # 0, 1

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLO(model='/mnt/hdd_4A/choemj/2025winterlab/type_choemj/weights/yolo11n.pt')

model.train(data='yolov11.yaml', epochs=1000, batch=128, imgsz=640, project="weights/trained/")