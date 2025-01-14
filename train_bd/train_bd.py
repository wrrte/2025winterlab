from ultralytics.models import YOLOv10

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 0, 1

#nvitop -m compact --only-compute
#duf --style ascii

model = YOLOv10('../weights/bd_origin_2.pt')

'''
import torch
print(torch.cuda.is_available())  # CUDA 사용 가능 여부 확인
print(torch.cuda.device_count())  # 사용 가능한 GPU 개수 확인
print(torch.cuda.current_device())  # 현재 사용 중인 GPU 확인
print(torch.cuda.get_device_name(0))  # GPU 0의 이름 출력

# 명시적으로 GPU 0을 선택
device = torch.device("cuda:0")
model.to(device)
'''

model.train(data='yolov10.yaml', epochs=500, batch=128, imgsz=640)