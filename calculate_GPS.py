import numpy as np
import geopy.distance
import subprocess
import re
import glob
import os

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

def calculate_angle_and_distance(image_width, image_height, target_x, depth_map, ref_distance, FOV):
    cx = image_width // 2

    dx = target_x - cx

    theta = (dx / (image_width / 2)) * (FOV / 2)

    target_x_int = int(target_x)
    target_y_int = int(image_height // 2)
    absolute_distance = depth_map[target_y_int, target_x_int] * ref_distance

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
    # run_dpt.py 실행
    result = subprocess.run(['python', 'run_dpt.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print("run_dpt.py 실행 중 오류가 발생했습니다.")
        print(result.stderr)
        exit(1)
    
    # PFM 폴더에서 가장 최근에 생성된 PFM 파일 찾기
    pfm_files = [f for f in glob.glob(os.path.join(pfm_folder_path, '*.pfm')) if f.endswith('.pfm')]
    if not pfm_files:
        print(f"PFM 파일을 찾을 수 없습니다: {pfm_folder_path}")
        exit(1)

    latest_pfm_file = max(pfm_files, key=os.path.getctime)  # 가장 최근에 수정된 파일
    print(f"가장 최근에 생성된 PFM 파일: {latest_pfm_file}")

    return latest_pfm_file

def get_latest_reference_point(record_dir):
    txt_files = [f for f in glob.glob(os.path.join(record_dir, '*.txt')) if f.endswith('.txt')]
    if not txt_files:
        print(f"참조 지점을 찾을 수 없습니다: {record_dir}")
        return None

    latest_txt_file = max(txt_files, key=os.path.getctime)
    print(f"가장 최근에 생성된 텍스트 파일: {latest_txt_file}")

    with open(latest_txt_file, 'r') as file:
        line = file.readline()
        match = re.search(r"5m point at \((\d+), (\d+)\)", line)
        if match:
            reference_x, reference_y = map(int, match.groups())
            return (reference_x, reference_y)
    return None

def predict_detection_points_gps(detection_script_path, pfm_folder_path, record_dir, current_gps, heading, ref_distance, FOV):
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

# 예제 사용
current_gps = (36.370077, 127.379437)  # 현재 GPS 위치 (위도, 경도)
heading = 180  # 카메라의 현재 바라보는 방향 (서쪽 85도)
reference_distance = 2.5  # 참조 지점의 실제 거리 (미터)
FOV = 72  # 화각 72도
pfm_folder_path = 'dpt_output/'  # PFM 파일이 저장된 폴더 경로
detection_script_path = 'run_bd.py'  # building detection 파일 경로
record_dir = 'roadview/reference_point'  # 텍스트 파일이 저장된 폴더 경로

predicted_gps_points = predict_detection_points_gps(detection_script_path, pfm_folder_path, record_dir, current_gps, heading, reference_distance, FOV)

# 결과 출력
for i, (gps, angle, distance) in enumerate(predicted_gps_points):
    print(f"Detection point {i+1}의 예측 GPS 좌표: {gps}, 각도: {angle:.2f}도, 거리: {distance:.2f}미터")