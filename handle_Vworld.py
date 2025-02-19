import requests
import cv2
import subprocess
import ast
import requests
import glob
import os
from geopy.distance import geodesic
from PIL import Image, ImageDraw, ImageFont
from dpt_type.GPS_dpt import GPS_dpt
from type_stereo/run_stereo import main as GPS_stereo


predict_type = "dpt"
#predict_type = "stereo"

# gps_est.py 스크립트를 실행하고 출력을 캡처
print(f"Running {predict_type}\n")

run

if predict_type == "dpt":
    While True:
        image, detection_points, predicted_gps_points = GPS_dpt()
else:
    While True:
        result = GPS_stereo()

# 출력된 내용을 문자열로 가져옴
output = result.stdout
print("Captured output from calculate_GPS.py:\n", output)

# 출력 내용에서 GPS 좌표 부분만 추출
lines = output.splitlines()
gps_coordinates = []
responses = []


print("\nExtracting GPS coordinates from output...")
for line in lines:
    if "predicted GPS coordinates" in line:
        # 좌표 부분만 추출
        start_idx = line.find(": (") + 3  # ": (" 다음부터 시작
        end_idx = line.find(")", start_idx)
        coordinate_str = line[start_idx:end_idx]
        gps_coordinate = ast.literal_eval(coordinate_str)
        gps_coordinates.append(gps_coordinate)
        print(f"Extracted GPS coordinate: {gps_coordinate}")


API_KEY = "3B8488FF-278C-3D24-992A-09B987D1CAB1"

url = "https://api.vworld.kr/req/address"


gps_coordinates.append([37.443108, 126.7143842])


for idx, gps_coordinate in enumerate(gps_coordinates):
    # 위도와 경도를 그대로 사용
    latitude = gps_coordinate[0]
    longitude = gps_coordinate[1]

    params = {
        "service": "address",
        "request": "getAddress",
        "format": "json",
        "crs": "epsg:4326",
        "point": f"{longitude},{latitude}",
        "type": "ROAD",
        "key": API_KEY
    }

    response = requests.post(url, data=params)
    if response.status_code == 200:
        try:
            address = response.json()['response']['result'][0]['text']
            print(address)
        except:
            print(response.json())