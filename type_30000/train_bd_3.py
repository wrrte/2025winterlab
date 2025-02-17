from ultralytics.models import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3" # 0, 1

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLO(model='best.pt')

model.train(data='yolov11_3.yaml', epochs=2000, batch=32, imgsz=640, project='/mnt/hdd_4A/choemj/2025winterlab/type_30000/weight')