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
    ref_depth_value = depth_map[ref_point[1], ref_point[0]]
    max_depth_value = np.max(depth_map)
    target_depth_value = depth_map[int(target_point[1]), int(target_point[0])]
    
    ratio = ref_distance / (ref_depth_value - max_depth_value)
    absolute_distance = np.abs(((max_depth_value - target_depth_value) * ratio))**3
    
    return absolute_distance

def calculate_angle_and_distance(image_width, image_height, target_x, target_y, depth_map, ref_distance, ref_point, FOV):

    s = time.time()
    
    absolute_distance = calculate_absolute_distance_for_point(
        depth_map, 
        ref_distance, 
        ref_point, 
        (target_x, target_y)
    )

    e = time.time()
    
    return e-s

def calculate_gps_coordinates(current_gps, heading, angle, distance):
    actual_angle = (heading + angle) % 360
    origin = geopy.Point(current_gps[0], current_gps[1])
    destination = geopy.distance.distance(meters=distance).destination(origin, actual_angle)
    return destination.latitude, destination.longitude

def get_detection_points_from_file(result_file):
    with open(result_file, 'r') as f:
        content = f.read()
    
    detection_points = []
    pattern = r'Center \(x, y\): \(([\d.]+), ([\d.]+)\)'
    matches = re.findall(pattern, content)

    for match in matches:
        x, y = map(float, match)
        detection_points.append((x, y))

    return detection_points

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

def process_image_set(image_base_name, pfm_folder_path, bb_results_folder, record_dir, current_gps, heading, ref_distance, FOV):
    ref_point = get_latest_reference_point(record_dir)
    if ref_point is None:
        print(f"Reference point not found for {image_base_name}")
        return []

    # Construct file paths
    pfm_file = os.path.join(pfm_folder_path, f"{image_base_name}.pfm")
    bb_result_file = os.path.join(bb_results_folder, f"{image_base_name}.txt")

    # Check if both files exist
    if not (os.path.exists(pfm_file) and os.path.exists(bb_result_file)):
        print(f"Missing files for {image_base_name}")
        return []

    # Load depth map
    depth_map, scale = load_pfm(pfm_file)
    height, width = depth_map.shape[:2]
    
    # Get detection points from BB result file
    detection_points = get_detection_points_from_file(bb_result_file)
    
    tt=0
    for point in detection_points:
        target_x, target_y = point
        
        tt += calculate_angle_and_distance(
            width, height, target_x, target_y, 
            depth_map, ref_distance, ref_point, FOV
        )



    return tt/len(detection_points)

def process_all_images(pfm_folder_path, bb_results_folder, record_dir, current_gps, heading, ref_distance, FOV):
    # Get all PFM files
    pfm_files = glob.glob(os.path.join(pfm_folder_path, '*.pfm'))
    all_results = []

    for pfm_file in pfm_files:
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(pfm_file))[0]
        
        # Process this image set
        all_results.append(process_image_set(
            base_name,
            pfm_folder_path,
            bb_results_folder,
            record_dir,
            current_gps,
            heading,
            ref_distance,
            FOV
        ))

    return all_results

# Example usage
current_gps = (36.370077, 127.379437)  # Current GPS position
heading = 180  # Camera heading
reference_distance = 2.5  # Reference point distance (meters)
FOV = 72  # Field of view
pfm_folder_path = 'dpt_output/'  # PFM files folder
bb_results_folder = 'bb_output/coordinate'  # Bounding box results folder
record_dir = 'roadview/reference_point'  # Reference point folder

# Process all images
results = process_all_images(
    pfm_folder_path,
    bb_results_folder,
    record_dir,
    current_gps,
    heading,
    reference_distance,
    FOV
)

print(results)
print(sum(results)/len(results))