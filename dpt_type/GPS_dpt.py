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



def calculate_absolute_distance_for_point(depth_map, ref_distance, ref_point, target_point):
    # Get depth values for reference and target points
    ref_depth_value = depth_map[ref_point[1], ref_point[0]]
    max_depth_value = np.max(depth_map)
    target_depth_value = depth_map[int(target_point[1]), int(target_point[0])]
    
    # Calculate ratio using reference point
    ratio = ref_distance / (ref_depth_value - max_depth_value)
    
    # Calculate absolute distance for target point
    absolute_distance = np.abs(((max_depth_value - target_depth_value) * ratio))**3
    
    return absolute_distance

def calculate_angle_and_distance(image_width, image_height, target_x, target_y, depth_map, ref_distance, ref_point, FOV):
    cx = image_width // 2
    dx = target_x - cx
    theta = (dx / (image_width / 2)) * (FOV / 2)
    
    absolute_distance = calculate_absolute_distance_for_point(
        depth_map, 
        ref_distance, 
        ref_point, 
        (target_x, target_y)
    )
    
    return theta, absolute_distance

def calculate_gps_coordinates(current_gps, heading, angle, distance):
    actual_angle = (heading + angle) % 360
    origin = geopy.Point(current_gps[0], current_gps[1])
    destination = geopy.distance.distance(meters=distance).destination(origin, actual_angle)
    return destination.latitude, destination.longitude

def get_latest_reference_point(record_dir):
    txt_files = [f for f in glob.glob(os.path.join(record_dir, 'reference_point', '*.txt'))]
    if not txt_files:
        print(f"No reference points found in: {record_dir}")
        return None

    latest_txt_file = max(txt_files, key=os.path.getctime)
    print(f"Most recent text file: {latest_txt_file}")

    with open(latest_txt_file, 'r') as file:
        line = file.readline()
        match = re.search(r"5m point at \((\d+), (\d+)\)", line)
        if match:
            reference_x, reference_y = map(int, match.groups())
            return (reference_x, reference_y)
    return None


def dpt_process(q, image_file):
    depth_map = run_dpt(image_file)  # DPT 실행
    q.put(depth_map)

def bd_process(q, image_file):
    model=YOLO("/mnt/hdd_4A/choemj/2025winterlab/type_30000/weight/train/weights/best.pt")
    detection_points = run_bd(image_file, model)
    q.put(detection_points)


def GPS_dpt(record_dir, current_gps, heading, ref_distance, FOV):
    global inference_time
    
    ref_point = get_latest_reference_point(record_dir)
    if ref_point is None:
        print("Reference point not found.")
        return []
    
    pngs = [f for f in glob.glob(os.path.join(record_dir, 'image', '*.png'))]
    image_file = max(pngs, key=os.path.getctime)

    q = mp.Queue()

    # 병렬 프로세스 실행
    p1 = mp.Process(target=dpt_process, args=(q, image_file))
    p2 = mp.Process(target=bd_process, args=(q, image_file))

    p1.start()
    p2.start()

    # 결과 가져오기
    depth_map = q.get()
    detection_points = q.get()

    p1.join()
    p2.join()

    height, width = depth_map.shape[:2]
    
    predicted_gps_points = []
    for point in detection_points:
        target_x, target_y = point
        
        angle, distance = calculate_angle_and_distance(
            width, height, target_x, target_y, 
            depth_map, ref_distance, ref_point, FOV
        )

        predicted_gps = calculate_gps_coordinates(current_gps, heading, angle, distance)
        predicted_gps_points.append((predicted_gps, angle, distance))

    return predicted_gps_points

# Example usage
current_gps = (37.5665, 126.9780)  # Current GPS position (latitude, longitude)
heading = 180  # Camera heading (180 degrees)
reference_distance = 2.5  # Reference point actual distance (meters)
FOV = 72  # Field of view 72 degrees
record_dir = 'roadview/'  # Text files folder path

inference_time = []

#mp.set_start_method('spawn')  # GPU 자원 공유를 위해 spawn 방식 사용

predicted_gps_points = GPS_dpt(
    record_dir,
    current_gps, heading, reference_distance, FOV
)

# Print results
for i, (gps, angle, distance) in enumerate(predicted_gps_points):
    print(f"Detection point {i+1} predicted GPS coordinates: {gps}, angle: {angle:.2f}°, distance: {distance:.2f}m")