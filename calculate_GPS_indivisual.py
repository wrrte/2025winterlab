import numpy as np
import math
import geopy.distance
import subprocess
import re
import glob
import os
import time

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
        
        return np.reshape(data, shape), scale

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

def run_detection_script(detection_script_path):
    result = subprocess.run(['python', detection_script_path], capture_output=True, text=True)
    output = result.stdout

    detection_points = []
    pattern = r'Center \(x, y\): \(([\d.]+), ([\d.]+)\)'
    matches = re.findall(pattern, output)

    for match in matches:
        x, y = map(float, match)
        detection_points.append((x, y))

    return detection_points

def run_dpt_and_get_latest_pfm(pfm_folder_path):
    # Run run_dpt.py
    result = subprocess.run(['python', 'run_dpt_test_cal.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running run_dpt.py")
        print(result.stderr)
        exit(1)

    # Find most recent PFM file
    pfm_files = [f for f in glob.glob(os.path.join(pfm_folder_path, '*.pfm'))]
    if not pfm_files:
        print(f"No PFM files found in: {pfm_folder_path}")
        exit(1)

    latest_pfm_file = max(pfm_files, key=os.path.getctime)
    print(f"Most recent PFM file: {latest_pfm_file}")

    return latest_pfm_file

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

def predict_detection_points_gps(detection_script_path, pfm_folder_path, record_dir, current_gps, heading, ref_distance, FOV):
    global inference_time
    
    ref_point = get_latest_reference_point(record_dir)
    if ref_point is None:
        print("Reference point not found.")
        return []

    pfm_file_path = run_dpt_and_get_latest_pfm(pfm_folder_path)

    depth_map, scale = load_pfm(pfm_file_path)
    height, width = depth_map.shape[:2]
    
    detection_points = run_detection_script(detection_script_path)
    
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
current_gps = (36.370077, 127.379437)  # Current GPS position (latitude, longitude)
heading = 180  # Camera heading (180 degrees)
reference_distance = 2.5  # Reference point actual distance (meters)
FOV = 72  # Field of view 72 degrees
pfm_folder_path = 'dpt_output_test_cal/'  # PFM files folder path
detection_script_path = 'run_bd.py'  # Building detection script path
record_dir = 'roadview/reference_point'  # Text files folder path

inference_time = []

predicted_gps_points = predict_detection_points_gps(
    detection_script_path, pfm_folder_path, record_dir,
    current_gps, heading, reference_distance, FOV
)

# Print results
for i, (gps, angle, distance) in enumerate(predicted_gps_points):
    print(f"Detection point {i+1} predicted GPS coordinates: {gps}, angle: {angle:.2f}°, distance: {distance:.2f}m")