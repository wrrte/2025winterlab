import cv2
import subprocess
import ast
import requests
import glob
import os
from geopy.distance import geodesic
from PIL import Image, ImageDraw, ImageFont

# gps_est.py 스크립트를 실행하고 출력을 캡처
print("Running calculate_GPS.py to get estimated GPS coordinates...\n")
result = subprocess.run(["python3", "calculate_GPS.py"], capture_output=True, text=True)

# 출력된 내용을 문자열로 가져옴
output = result.stdout
print("Captured output from calculate_GPS.py:\n", output)

# 출력 내용에서 GPS 좌표 부분만 추출
lines = output.splitlines()
gps_coordinates = []
responses = []

print("\nExtracting GPS coordinates from output...")
for line in lines:
    if "예측 GPS 좌표" in line:
        # 좌표 부분만 추출
        start_idx = line.find(": (") + 3  # ": (" 다음부터 시작
        end_idx = line.find(")", start_idx)
        coordinate_str = line[start_idx:end_idx]
        gps_coordinate = ast.literal_eval(coordinate_str)
        gps_coordinates.append(gps_coordinate)
        print(f"Extracted GPS coordinate: {gps_coordinate}")

# API 엔드포인트 (좌표로 도로명 주소 조회)
coordinate_url = "https://gs1geo.oliot.kr/api/address"

print("\nSending API requests to get road names...")
# 모든 추출된 좌표에 대해 API 요청 보내기
for idx, gps_coordinate in enumerate(gps_coordinates):
    # 위도와 경도를 그대로 사용
    latitude = gps_coordinate[0]
    longitude = gps_coordinate[1]

    # 좌표를 문자열로 변환
    coordinate_str = f"{latitude}, {longitude}"

    # 파라미터 설정
    params = {
        'coordinate': coordinate_str
    }

    # GET 요청 보내기
    response = requests.get(coordinate_url, params=params)

    # 응답 상태 코드 확인 및 응답 데이터 저장
    if response.status_code == 200:
        # 응답 JSON 데이터 가져오기
        data = response.json()
        if data['status']:
            responses.append({
                "index": idx + 1,
                "gps_coordinate": gps_coordinate,
                "road_name": data['data']['road_name'],
                "raw_data": data
            })
            print(f"Received road name '{data['data']['road_name']}' for coordinate {gps_coordinate}")
        else:
            print(f"Failed to get road name for coordinate {gps_coordinate}")
    else:
        print(f"API request failed for coordinate {gps_coordinate} with status code {response.status_code}")

# 동일한 도로명 주소에 대한 응답 그룹화
road_name_groups = {}
print("\nGrouping responses by road name...")
for response in responses:
    road_name = response['road_name']
    if road_name not in road_name_groups:
        road_name_groups[road_name] = []
    road_name_groups[road_name].append(response)
    print(f"Grouped coordinate {response['gps_coordinate']} under road name '{road_name}'")

# 결과 리스트에 대해 최종 필터링
final_results = []
print("\nFiltering results to keep the closest coordinates for duplicated road names...")
for road_name, grouped_responses in road_name_groups.items():
    if len(grouped_responses) > 1:
        print(f"\nProcessing duplicated road name '{road_name}' with {len(grouped_responses)} coordinates...")
        # 도로명 주소에 대한 정확한 좌표를 가져오기 위해 API 호출
        road_name_url = "https://gs1geo.oliot.kr/api/address"
        road_name_params = {'roadName': road_name}
        correct_coord_response = requests.get(road_name_url, params=road_name_params)

        if correct_coord_response.status_code == 200:
            correct_data = correct_coord_response.json()
            if correct_data['status']:
                correct_coord = correct_data['data']['coordinate']
                print(f"Correct coordinate for road name '{road_name}': {correct_coord}")
                min_distance = float('inf')
                closest_response = None
                for response in grouped_responses:
                    estimated_coord = response['gps_coordinate']
                    distance = geodesic(estimated_coord, correct_coord).meters
                    print(f"Distance from {estimated_coord} to correct coordinate: {distance:.2f} meters")
                    if distance < min_distance:
                        min_distance = distance
                        closest_response = response
                        print(f"New closest coordinate found: {estimated_coord} with distance {min_distance:.2f} meters")
                # 모든 그룹의 가장 가까운 좌표를 결과에 추가
                final_results.append(closest_response)
            else:
                print(f"Failed to get correct coordinate for road name '{road_name}' - API status was False")
        else:
            print(f"Failed to get correct coordinate for road name '{road_name}' with status code {correct_coord_response.status_code}")
    else:
        # 중복되지 않는 경우 그대로 결과에 추가
        final_results.extend(grouped_responses)

# 최종 결과 출력
print("\nFinal results:")
for result in final_results:
    print(f"Final GPS coordinate {result['index']}: Estimated - {result['gps_coordinate']}, Road Name - {result['road_name']}, Full Response - {result['raw_data']}")

# 최종 결과 이미지 시각화
print("\nVisualizing final results...")

# `visualize` 폴더에 있는 이미지 파일 찾기
visualize_images = glob.glob('visualize/*.png')
if not visualize_images:
    print("No images found in the visualize folder.")
    exit(1)

# 최종 시각화 이미지 폴더 설정
final_visualize_folder = 'final_visualize'
os.makedirs(final_visualize_folder, exist_ok=True)

# 한글 폰트를 사용할 수 있도록 PIL을 통해 글꼴 설정
font_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"  # 시스템에 설치된 폰트 경로
font = ImageFont.truetype(font_path, 20)

# 각 시각화된 이미지에 텍스트 추가
for result in final_results:
    gps_coord = result['gps_coordinate']
    road_name = result['road_name']
    sgln = result['raw_data']['data'].get('sgln', 'N/A')  # SGLN 정보가 없는 경우 'N/A'로 대체
    index = result['index']

    img_path = visualize_images[index - 1]  # 인덱스에 해당하는 이미지 선택
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    # 바운딩 박스 정보 가져오기
    bbox_file_path = 'upgradebb_output/new_result.txt'
    with open(bbox_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            bbox_data = line.strip().split()
            if len(bbox_data) < 6:
                print(f"Warning: Expected at least 6 values but got {len(bbox_data)}: {bbox_data}")
                continue

            # 첫 번째 클래스 요소는 무시하고 나머지 요소만 사용
            conf, x_min, y_min, x_max, y_max = map(float, bbox_data[1:])
            
            # 이미지 위에 텍스트 추가 (각 정보 한 줄씩)
            text_1 = f"GPS: {gps_coord}"
            text_2 = f"Road: {road_name}"
            text_3 = f"SGLN: {sgln}"
            text_position = (int(x_min * img.width), max(int(y_min * img.height) - 30, 20))  # 텍스트가 이미지 바깥으로 나가지 않도록 최소 20 픽셀 유지

            # 검정색 텍스트로 추가
            draw.text(text_position, text_1, font=font, fill=(0, 0, 0))
            draw.text((text_position[0], text_position[1] + 20), text_2, font=font, fill=(0, 0, 0))
            draw.text((text_position[0], text_position[1] + 40), text_3, font=font, fill=(0, 0, 0))

    # 최종 결과 이미지를 final_visualize 폴더에 저장
    final_image_path = os.path.join(final_visualize_folder, f'final_result_visualized_{index}.png')
    img.save(final_image_path)
    print(f"Final result image saved to {final_image_path}")