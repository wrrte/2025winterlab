import numpy as np
import geopy.distance
import subprocess
import glob
import os


def load_txt(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            data.append(values)
    return data

def calculate_gps_coordinates(current_gps, heading, angle, distance):
    actual_angle = (heading + angle) % 360
    origin = geopy.Point(current_gps[0], current_gps[1])
    destination = geopy.distance.distance(meters=distance).destination(origin, actual_angle)
    return destination.latitude, destination.longitude

def run_stereo_model():
    result = subprocess.run(['python', 'run_stereo.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running run_stereo.py")
        exit(1)

def fetch_latest_results(image_folder_path, coord_folder_path, depth_folder_path):
    image_file_path = max(glob.glob(os.path.join(image_folder_path, '*.png')), key=os.path.getctime)
    coord_file_path = max(glob.glob(os.path.join(coord_folder_path, '*.txt')), key=os.path.getctime)
    depth_file_path = max(glob.glob(os.path.join(depth_folder_path, '*.txt')), key=os.path.getctime)
    return image_file_path, coord_file_path, depth_file_path

def predict_gps(image_folder_path, coord_folder_path, depth_folder_path, current_gps, heading):
    image_file, coord_file, depth_file = fetch_latest_results(image_folder_path, coord_folder_path, depth_folder_path)
    coordinates = load_txt(coord_file)
    depths = load_txt(depth_file)

    if len(coordinates) != len(depths):
        raise ValueError(f"number of detection mismatch")

    gps_results = []
    for (x_min, y_min, x_max, y_max), (depth,) in zip(coordinates, depths):
        target_x = (x_min + x_max) / 2

        angle = (target_x - 320) / 320 * 35  # Assuming 640px width and 70-degree FOV
        gps_coordinates = calculate_gps_coordinates(current_gps, heading, angle, depth)
        gps_results.append((gps_coordinates, (x_min, y_min, x_max, y_max), depth))
    
    return image_file, gps_results


def main():
    current_gps = (37.5665, 126.9780) # Replace with real gps
    heading = 180 # Replace with real heading
    
    image_folder_path = "/home/seyeon/2025winterlab/type_stereo/test/test_results/images" 
    coord_folder_path = "/home/seyeon/2025winterlab/type_stereo/test/test_results/coordinates"
    depth_folder_path = "/home/seyeon/2025winterlab/type_stereo/test/test_results/depth"


    run_stereo_model()
    image_file, predicted_gps = predict_gps(image_folder_path, coord_folder_path, depth_folder_path, current_gps, heading)
    
    print(f"Image File: {image_file}")
    for i, (gps, bbox, distance) in enumerate(predicted_gps):
        print(f"Detection {i+1}: GPS = {gps}, BBox = {bbox}, Distance = {distance:.2f}m")

if __name__ == "__main__":
    main()
