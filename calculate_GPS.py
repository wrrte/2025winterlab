import numpy as np
import math
import geopy.distance
import subprocess
import re
import glob
import os
import argparse

def load_pfm(file_path):
    with open(file_path, "rb") as f:
        header = f.readline().decode('latin-1').rstrip()
        color = header == 'PF' # true 면 colored / false 면 grayscale
        
        dims = f.readline().decode('latin-1')
        width, height = map(int, dims.split())
        
        scale = float(f.readline().decode('latin-1').rstrip())
        endian = '<' if scale < 0 else '>'
        scale  = abs(scale)
        
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        
        return np.reshape(data, shape), scale

def load_txt(file_path):
    depth_data = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split(',')))
            depth_data.append(values)
    
    return depth_data

def calculate_gps_coordinates(current_gps, heading, angle, distance):
    actual_angle = (heading + angle) % 360
    origin = geopy.Point(current_gps[0], current_gps[1])
    destination = geopy.distance.distance(meters=distance).destination(origin, actual_angle)
    return destination.latitude, destination.longitude

def run_detection_script(detection_script_path):
    result = subprocess.run(['python', detection_script_path], capture_output=True, text=True)
    output = result.stdout
    pattern = r'Center \(x, y\): \(([\d.]+), ([\d.]+)\)'
    matches = re.findall(pattern, output)

    return [(float(x), float(y)) for x, y in matches]

def predict_gps_from_type1(detection_script_path, pfm_folder_path, current_gps, heading, ref_distance, FOV):
    pfm_file_path = max(glob.glob(os.path.join(pfm_folder_path, '*.pfm')), key=os.path.getctime)
    depth_map, scale = load_pfm(pfm_file_path)
    height, width = depth_map.shape[:2]
    detection_points = run_detection_script(detection_script_path)

    gps_results = []
    for target_x, target_y in detection_points:
        dx = target_x - (width // 2)
        angle = (dx / (width / 2)) * (FOV / 2)
        distance = depth_map[int(target_y), int(target_x)] * ref_distance
        gps_results.append((calculate_gps_coordinates(current_gps, heading, angle, distance), angle, distance))
    
    return gps_results

def predict_gps_from_type2(txt_folder_path, current_gps, heading):
    txt_file_path = max(glob.glob(os.path.join(txt_folder_path, '*.txt')), key=os.path.getctime)
    depth_data = load_txt(txt_file_path)

    gps_results = []
    for entry in depth_data:
        target_x, target_y, depth = entry
        angle = (target_x - 320) / 320 * 35 # 640px width, 70 FOV 일 경우
        gps_results.append((calculate_gps_coordinates(current_gps, heading, angle, depth), angle, depth))
    
    return gps_results

def main():
    parser = argparse.ArgumentParser(description="Predict GPS coordinates using depth estimation.")
    parser.add_argument('--type', choices=['1', '2'], required=True, help="Choose type of depth estimation: 1 for DPT Model, 2 for Stereo Camera")
    parser.add_argument('--detection_script', type=str, default='run_bd.py', help="Path to the detection script")
    parser.add_argument('--pfm_folder', type=str, default='dpt_output/', help="Path to PFM files folder (for Type 1)")
    parser.add_argument('--txt_folder', type=str, default='stereo_output/', help="Path to TXT files folder (for Type 2)")
    parser.add_argument('--latitude', type=float, required=True, help="Current GPS latitude")
    parser.add_argument('--longitude', type=float, required=True, help="Current GPS longitude")
    parser.add_argument('--heading', type=float, required=True, help="Camera heading direction")
    parser.add_argument('--reference_distance', type=float, default=2.5, help="Reference distance for Type 1 (meters)")
    parser.add_argument('--fov', type=float, default=72, help="Camera field of view (degrees)")
    
    args = parser.parse_args()
    current_gps = (args.latitude, args.longitude)
    
    if args.type == '1':
        predicted_gps = predict_gps_from_type1(args.detection_script, args.pfm_folder, current_gps, args.heading, args.reference_distance, args.fov)
    else:
        predicted_gps = predict_gps_from_type2(args.txt_folder, current_gps, args.heading)
    
    for i, (gps, angle, distance) in enumerate(predicted_gps):
        print(f"Detection {i+1}: GPS = {gps}, Angle = {angle:.2f}°, Distance = {distance:.2f}m")

if __name__ == "__main__":
    main()