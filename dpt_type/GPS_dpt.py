import numpy as np
import geopy.distance
import glob
import os
from run_dpt import run_dpt
from run_bd import run_bd
from ultralytics.models import YOLO
import cv2
import torch

import torch.multiprocessing as mp

os.environ["CUDA_VISIBLE_DEVICES"]="3" # 0, 1

def calculate_angle_and_distance(depth_map, image_width, max_depth_value, ratio, target_point, FOV):
    
    cx = image_width // 2
    dx = target_point[0] - cx
    theta = (dx / (image_width / 2)) * (FOV / 2)
    
    target_depth_value = depth_map[int(target_point[1]), int(target_point[0])]
    absolute_distance = np.abs(((max_depth_value - target_depth_value) * ratio))**1.6
    
    return theta, absolute_distance

def calculate_gps_coordinates(current_gps, heading, angle, distance):
    actual_angle = (heading + angle) % 360
    origin = geopy.Point(current_gps[0], current_gps[1])
    destination = geopy.distance.distance(meters=distance).destination(origin, actual_angle)
    return destination.latitude, destination.longitude

def process_results(result_queue, func, *args):
    result = func(*args)
    result_queue.put(result)

def GPS_dpt(model_path, cap, current_gps, heading, ref_distance, FOV):

    model=YOLO(model_path)

    ret, image = cap.read()

    #'''
    depth_map = run_dpt(image) #모델 바꾸면 타입도 수정해야해!
    model=YOLO(model_path)
    detection_points = run_bd(image, model)
    #'''

    height, width = depth_map.shape[:2]

    ref_depth_value = depth_map[int(height*7/8), int(width/2)]
    max_depth_value = np.max(depth_map)
    
    # Calculate ratio using reference point
    ratio = ref_distance / (max_depth_value - ref_depth_value)
    
    predicted_gps_points = []
    for x_min, y_min, x_max, y_max in detection_points:

        point = ((x_min+x_max)/2, (y_min+y_max)/2)
        
        angle, distance = calculate_angle_and_distance(
            depth_map, width, max_depth_value, ratio, point, FOV)

        predicted_gps = calculate_gps_coordinates(current_gps, heading, angle, distance)
        predicted_gps_points.append(predicted_gps)

    return image, detection_points, predicted_gps_points

if __name__ == "__main__":

    print(torch.__version__)

    # Example usage
    current_gps = (37.5665, 126.9780)  # Current GPS position (latitude, longitude)
    heading = 180  # Camera heading (180 degrees)
    reference_distance = 10.4054  # Reference point actual distance (meters)
    FOV = 72  # Field of view 72 degrees
    record_dir = 'roadview/left'  # Text files folder path
    model_path = "/mnt/hdd_4A/choemj/2025winterlab/type_30000/weight/train/weights/best.pt"

    #mp.set_start_method('spawn')  # GPU 자원 공유를 위해 spawn 방식 사용

    cap = cv2.VideoCapture(0)
    print(cap.isOpened())

    
    while True:
        image, detection_points, predicted_gps_points = GPS_dpt(
            model_path, cap, current_gps, heading, reference_distance, FOV)
        


