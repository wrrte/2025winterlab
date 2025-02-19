import numpy as np
import math
import geopy.distance
import re
import glob
import os
import time
from run_dpt import run_dpt
from run_bd import run_bd
from ultralytics.models import YOLO

import torch.multiprocessing as mp

os.environ["CUDA_VISIBLE_DEVICES"]="3" # 0, 1

def calculate_angle_and_distance(depth_map, image_width, max_depth_value, ratio, target_point, FOV):
    
    cx = image_width // 2
    dx = target_point[0] - cx
    theta = (dx / (image_width / 2)) * (FOV / 2)
    
    target_depth_value = depth_map[int(target_point[1]), int(target_point[0])]
    absolute_distance = np.abs(((max_depth_value - target_depth_value) * ratio))**3
    
    return theta, absolute_distance

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


def GPS_dpt(record_dir, current_gps, heading, ref_distance, FOV):

    
    
    pngs = [f for f in glob.glob(os.path.join(record_dir, '*.png'))]
    image_file = max(pngs, key=os.path.getctime)

    depth_map = run_dpt(image_file)
    model=YOLO("/mnt/hdd_4A/choemj/2025winterlab/type_30000/weight/train/weights/best.pt")
    detection_points = run_bd(image_file, model)

    '''
    q = mp.Queue()

    # 병렬 프로세스 실행
    p1 = mp.Process(target=dpt_process, args=(q, image_file))
    p2 = mp.Process(target=bd_process, args=(q, image_file, model))

    p1.start()
    p2.start()

    # 결과 가져오기
    depth_map = q.get()
    detection_points = q.get()

    p1.join()
    p2.join()
    '''

    height, width = depth_map.shape[:2]

    ref_depth_value = depth_map[int(width/2), int(height*7/8)]
    max_depth_value = np.max(depth_map)
    
    # Calculate ratio using reference point
    ratio = ref_distance / (ref_depth_value - max_depth_value)
    
    predicted_gps_points = []
    for point in detection_points:
        
        angle, distance = calculate_angle_and_distance(
            depth_map, width, max_depth_value, ratio, point, FOV)

        predicted_gps = calculate_gps_coordinates(current_gps, heading, angle, distance)
        predicted_gps_points.append((predicted_gps, angle, distance))

    return predicted_gps_points

# Example usage
current_gps = (37.5665, 126.9780)  # Current GPS position (latitude, longitude)
heading = 180  # Camera heading (180 degrees)
reference_distance = 5.0  # Reference point actual distance (meters)
FOV = 72  # Field of view 72 degrees
record_dir = 'roadview/left'  # Text files folder path

#mp.set_start_method('spawn')  # GPU 자원 공유를 위해 spawn 방식 사용

predicted_gps_points = GPS_dpt(
    record_dir,
    current_gps, heading, reference_distance, FOV
)

# Print results
for i, (gps, angle, distance) in enumerate(predicted_gps_points):
    print(f"Detection point {i+1} predicted GPS coordinates: {gps}, angle: {angle:.2f}°, distance: {distance:.2f}m")