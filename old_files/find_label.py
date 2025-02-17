import cv2
import json
import numpy as np

def segmentation_to_bbox(segmentation):

    segmentation = [int(x) for x in segmentation]

    if not segmentation or len(segmentation) < 6:
        print("Error: 유효한 segmentation 데이터가 아닙니다.")
        return

    # x, y 좌표 분리
    x_coords = segmentation[0::2]  # 홀수 인덱스 (x 좌표)
    y_coords = segmentation[1::2]  # 짝수 인덱스 (y 좌표)

    # Bounding box 계산
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    return x_min, y_min, x_max, y_max

def draw_bounding_boxes(image_path, source_json_path, output_image_path, num):
    image = cv2.imread(image_path)
    
    with open(source_json_path) as ff:
        data = json.load(ff)

    if (data["images"]["height"] != image.shape[0]) or (data["images"]["width"] != image.shape[1]):
        print(f"{num}번 {filename} 파일 크기 이상함")
    

    for object in data["annotations"]:
        if object["category_id"] == num:
            
            x_min, y_min, x_max, y_max = segmentation_to_bbox(object["segmentation"][0])

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # 결과 이미지를 지정된 경로에 저장
    cv2.imwrite(output_image_path, image)

filename = "201213_E_14_CCW_in_E_B_000_00302"

image_path = f"datasets/train/images/{filename}.png"
json_path = f"datasets/train/annotations/{filename}_PGON.json"

#for num in range(1, 4):
num = 1
if True:
    output_path = f"/mnt/hdd_4A/choemj/2025winterlab/type_choemj/{num}.png"
    draw_bounding_boxes(image_path, json_path, output_path, num)
