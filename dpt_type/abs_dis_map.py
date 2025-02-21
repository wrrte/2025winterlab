import numpy as np
import geopy.distance
import re
import glob
import os
from run_dpt import run_dpt
from run_bd import run_bd
from ultralytics.models import YOLO
import util

import cv2
import matplotlib.pyplot as plt

import torch.multiprocessing as mp

os.environ["CUDA_VISIBLE_DEVICES"]="3" # 0, 1

def calculate_absolute_distance_for_point(depth_map, ref_distance):
    height, width = depth_map.shape[:2]

    # Use the provided reference point
    ref_depth_value = depth_map[int(height*7/8), int(width/2)]
    max_depth_value = np.max(depth_map)

    ratio = ref_distance / (max_depth_value - ref_depth_value)

    print("ratio :", ratio)

    absolute_distances = np.abs((max_depth_value-depth_map)*ratio)

    return absolute_distances

def calculate_angle_and_distance(image_width, image_height, depth_map, ref_distance, FOV):
    cx = image_width // 2
    
    absolute_distance = calculate_absolute_distance_for_point(
        depth_map, 
        ref_distance
    )
    
    return absolute_distance

def calculate_gps_coordinates(current_gps, heading, angle, distance):
    actual_angle = (heading + angle) % 360
    origin = geopy.Point(current_gps[0], current_gps[1])
    destination = geopy.distance.distance(meters=distance).destination(origin, actual_angle)
    return destination.latitude, destination.longitude

def dpt_process(q, image_file):
    depth_map = run_dpt(image_file)  # DPT 실행
    q.put(depth_map)

def bd_process(q, image_file):
    model=YOLO("/mnt/hdd_4A/choemj/2025winterlab/type_30000/weight/train/weights/best.pt")
    detection_points = run_bd(image_file, model)
    q.put(detection_points)


def GPS_dpt(record_dir, ref_distance, FOV):
    
    pngs = [f for f in glob.glob(os.path.join(record_dir, 'left', '*.png'))]
    for image_file in pngs:
        img = util.io.read_image(image_file)

        depth_map = run_dpt(img)
        model=YOLO("/mnt/hdd_4A/choemj/2025winterlab/type_30000/weight/train/weights/best.pt")


        height, width = depth_map.shape[:2]

        print(width, height)
        #ref_point = (ref_point[0], height-ref_point[1])
            
        distance = calculate_angle_and_distance(
                width, height,
                depth_map, ref_distance, FOV
            )

        if distance is None or distance.size == 0:
            print("Error: distance 데이터가 없습니다.")
            return
        
        print(np.min(distance), np.max(distance))
        
        distance = np.nan_to_num(distance, nan=0.0, posinf=255.0, neginf=0.0)

        min_val = np.min(distance)
        max_val = np.max(distance)

        #distance = (distance - min_val) * (255 / (max_val-min_val))


        # 2. distance 값을 0~255 범위로 정규화
        distance_norm = np.uint8(distance)  # 정수형 변환 (cv2 적용을 위해)

        # 3. 컬러맵 적용 및 시각화
        plt.figure(figsize=(8, 6))
        img = plt.imshow(distance_norm, cmap='plasma', interpolation='bilinear', aspect='auto')
        cbar = plt.colorbar(img)
        cbar.set_label("Distance (meters)", fontsize=12)
        
        # 4. 제목 및 축 설정
        plt.title("Absolute Distance Map", fontsize=14)
        plt.xlabel("X-axis (pixels)", fontsize=12)
        plt.ylabel("Y-axis (pixels)", fontsize=12)

        # 5. 이미지 저장
        plt.savefig(os.path.join("dpt_type/absolute", image_file), dpi=300, bbox_inches='tight')
        print(f"이미지가 '{image_file}'로 저장되었습니다.")

    
    return 



# Example usage
current_gps = (37.5665, 126.9780)  # Current GPS position (latitude, longitude)
heading = 180  # Camera heading (180 degrees)
reference_distance = 10.4054  # Reference point actual distance (meters)
FOV = 72  # Field of view 72 degrees
record_dir = 'roadview/'  # Text files folder path

#mp.set_start_method('spawn')  # GPU 자원 공유를 위해 spawn 방식 사용

GPS_dpt(record_dir, reference_distance, FOV)
