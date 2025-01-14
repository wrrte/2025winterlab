from ultralytics.models import YOLOv10

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # 0, 1

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLOv10('../weights/bd/bd_yolov10m.pt')

model.train(data='yolov10.yaml', epochs=500, batch=32, imgsz=640)