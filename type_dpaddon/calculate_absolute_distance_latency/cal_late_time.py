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

def calculate_absolute_distance(depth_map, ref_distance, ref_point):
    height, width = depth_map.shape[:2]

    # Use the provided reference point
    ref_depth_value = depth_map[ref_point[1], ref_point[0]]
    max_depth_value = np.max(depth_map)

    ratio = ref_distance / (ref_depth_value - max_depth_value)

    absolute_distances = np.abs(((max_depth_value - depth_map) * ratio))**3

    return absolute_distances

def run_dpt_and_get_latest_pfm(pfm_file_path):
    # run_dpt.py 실행
    result = subprocess.run(['python', 'run_dpt.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("run_dpt.py 실행 중 오류가 발생했습니다.")
        print(result.stderr)
        exit(1)

    pfm_file = glob.glob(pfm_file_path)

    return pfm_file

def get_latest_reference_point(record_dir):
    txt_files = [f for f in glob.glob(os.path.join(record_dir, '*.txt')) if f.endswith('.txt')]
    if not txt_files:
        print(f"참조 지점을 찾을 수 없습니다: {record_dir}")
        return None

    latest_txt_file = max(txt_files, key=os.path.getctime)
    #print(f"가장 최근에 생성된 텍스트 파일: {latest_txt_file}")

    with open(latest_txt_file, 'r') as file:
        line = file.readline()
        match = re.search(r"5m point at \((\d+), (\d+)\)", line)
        if match:
            reference_x, reference_y = map(int, match.groups())
            return (reference_x, reference_y)
    return None

def predict_detection_points_gps(pfm_file_path, record_dir, ref_distance):
    
    ref_point = get_latest_reference_point(record_dir)
    if ref_point is None:
        print("참조 지점을 찾을 수 없습니다.")
        return []

    start_time = time.time()

    depth_map, scale = load_pfm(pfm_file_path)
    height, width = depth_map.shape[:2]

    middle_time = time.time()

    absolute_distances = calculate_absolute_distance(depth_map, ref_distance, ref_point)

    end_time = time.time()
    tot_time = end_time - start_time

    return end_time - start_time, end_time - middle_time

# 예제 사용
reference_distance = 2.5  # 참조 지점의 실제 거리 (미터)
pfm_folder_path = 'dpt_output/'  # PFM 파일이 저장된 폴더 경로
record_dir = 'roadview/reference_point'  # 텍스트 파일이 저장된 폴더 경로

tot_times = []
cal_times = []

files = glob.glob(os.path.join(pfm_folder_path, '*.pfm'))

for pfm_file_path in files:
    tt, mt = predict_detection_points_gps(pfm_file_path, record_dir, reference_distance)
    tot_times.append(tt)
    cal_times.append(mt)


tsum = 0
csum = 0
for i in range(len(tot_times)):
    tsum += tot_times[i]
    csum += cal_times[i]
print(tot_times)
print(cal_times)
print(tsum/len(tot_times))
print(csum/len(tot_times))
