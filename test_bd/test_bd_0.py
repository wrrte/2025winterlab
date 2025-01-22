from ultralytics import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" # 0, 1

import time
import torch
import sys
import logging

original_stdout = sys.stdout
sys.stdout = open('/dev/null', 'w')
logging.getLogger().setLevel(logging.WARNING)  # WARNING 수준 이상만 출력

# Set the random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mts = ["l", "m", "s", "x"]
message = []

for model_type in mts:

    model = YOLO(model=f"runs/detect/train_{model_type}/weights/best.pt", verbose=False)
    #model.train(data='yolov10.yaml', epochs=500, batch=16, imgsz=640)
    test_images_dir = '/home/choemj/2025winterlab/train_bd/test/images/'

    inference_time = []
    fps = []

    for i in range (10):

        #fps 계산
        start_time = time.time()
        outputs = model(test_images_dir)
        end_time = time.time()
        tot_time = end_time - start_time

        inference_time.append(tot_time)
        num_images = len(os.listdir(test_images_dir))
        fps.append(num_images / tot_time)

    #for i in range(10):
    #    print(f"{i+1}th Inference Time: {inference_time[i]:.4f}s, {i+1}th FPS: {fps[i]:.4f}")

    sorted_fps = sorted(fps)
    trimmed_fps = sorted_fps[1:-1]
    avg_fps = sum(sorted_fps) / len(sorted_fps)

    sys.stdout = original_stdout
    message.append(f"type {model_type} Average FPS : {avg_fps:.4f}")
    sys.stdout = open('/dev/null', 'w')

for m in message:
    print(m)