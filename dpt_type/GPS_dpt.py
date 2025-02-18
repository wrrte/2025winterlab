import numpy as np
import math
import geopy.distance
import subprocess
import re
import glob
import os
import time

import torch.multiprocessing as mp

def load_pfm(file_path):
    with open(file_path, "rb") as f:
        header = f.readline().decode('latin-1').rstrip()
        color = header == 'PF'
        
        dims = f.readline().decode('latin-1')
        width, height = map(int, dims.split())
        
        scale = float(f.readline().decode('latin-1').rstrip())
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'
        
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        
        return np.reshape(data, shape)

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

def run_bd():
    result = subprocess.run(['python', 'dpt_type/run_bd.py'], capture_output=True, text=True)
    output = result.stdout

    detection_points = []
    pattern = r'Center \(x, y\): \(([\d.]+), ([\d.]+)\)'
    matches = re.findall(pattern, output)

    for match in matches:
        x, y = map(float, match)
        detection_points.append((x, y))

    return detection_points

def run_dpt():
    # Run run_dpt.py
    result = subprocess.run(['python', 'dpt_type/run_dpt.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running run_dpt.py")
        print(result.stderr)
        exit(1)

    # Find most recent PFM file
    pfm_files = [f for f in glob.glob(os.path.join("dpt_type/dpt_output/", '*.pfm'))]
    if not pfm_files:
        print(f"No PFM files found in: dpt_type/dpt_output/")
        exit(1)

    latest_pfm_file = max(pfm_files, key=os.path.getctime)
    print(f"Most recent PFM file: {latest_pfm_file}")

    return load_pfm(latest_pfm_file)

def get_latest_reference_point(record_dir):
    txt_files = [f for f in glob.glob(os.path.join(record_dir, '*.txt'))]
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


def GPS_dpt(pfm_folder_path, record_dir, current_gps, heading, ref_distance, FOV):
    global inference_time
    
    ref_point = get_latest_reference_point(record_dir)
    if ref_point is None:
        print("Reference point not found.")
        return []

    depth_map = run_dpt()
    detection_points = run_bd()

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
pfm_folder_path = 'dpt_type/dpt_output/'  # PFM files folder path
record_dir = 'roadview/reference_point'  # Text files folder path

inference_time = []

#mp.set_start_method('spawn')  # GPU 자원 공유를 위해 spawn 방식 사용

predicted_gps_points = GPS_dpt(
    pfm_folder_path, record_dir,
    current_gps, heading, reference_distance, FOV
)

# Print results
for i, (gps, angle, distance) in enumerate(predicted_gps_points):
    print(f"Detection point {i+1} predicted GPS coordinates: {gps}, angle: {angle:.2f}°, distance: {distance:.2f}m")