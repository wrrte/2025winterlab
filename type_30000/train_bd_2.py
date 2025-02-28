from ultralytics.models import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2" # 0, 1

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLO(model='/mnt/hdd_4A/choemj/2025winterlab/type_origin/weights/bd_trained/train_n/weights/best.pt')

model.train(data='yolov11.yaml', epochs=2000, batch=128, imgsz=640, project="weight")