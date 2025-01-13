from ultralytics.models import YOLOv10

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 0~7

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLOv10('../weights/bd_01131614.pt')

model.train(data='yolov10.yaml', epochs=500, batch=256, imgsz=640)