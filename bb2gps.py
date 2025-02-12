import numpy as np
import math
import geopy.distance
import subprocess
import re
import glob
import os

def bb2gps(detection_script_path, pfm_folder_path, record_dir, current_gps, heading, ref_distance, FOV):
    ref_point = get_latest_reference_point(record_dir)
    if ref_point is None:
        print("참조 지점을 찾을 수 없습니다.")
        return []

    pfm_file_path = run_dpt_and_get_latest_pfm(pfm_folder_path)
    depth_map, scale = load_pfm(pfm_file_path)
    height, width = depth_map.shape[:2]

    absolute_distances = calculate_absolute_distance(depth_map, ref_distance, ref_point)

    detection_points = run_detection_script(detection_script_path)

    predicted_gps_points = []

    for point in detection_points:
        target_x, target_y = point

        angle, distance = calculate_angle_and_distance(width, height, target_x, absolute_distances, ref_distance, FOV)

        predicted_gps = calculate_gps_coordinates(current_gps, heading, angle, distance)
        predicted_gps_points.append((predicted_gps, angle, distance))

    return predicted_gps_points

predicted_gps_points = bb2gps(detection_script_path, pfm_folder_path, record_dir, current_gps, heading, reference_distance, FOV)

# 결과 출력
for i, (gps, angle, distance) in enumerate(predicted_gps_points):
    print(f"Detection point {i+1}의 예측 GPS 좌표: {gps}, 각도: {angle:.2f}도, 거리: {distance:.2f}미터")