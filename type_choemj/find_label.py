import cv2
import json
import numpy as np

def draw_bounding_boxes(image_path, source_json_path, output_image_path, num):
    image = cv2.imread(image_path)
    
    with open(source_json_path) as ff:
        data = json.load(ff)

    if (data["images"]["height"] != image.shape[0]) or (data["images"]["width"] != image.shape[1]):
        print(f"{num}번 {filename} 파일 크기 이상함")
    

    for object in data["annotations"]:
        if object["category_id"] == num:
            
            #x_min, y_min, x_max, y_max = object["bbox"]

            points = np.array(object["segmentation"], dtype=np.int32).reshape((-1, 1, 2))

            

            cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    # 결과 이미지를 지정된 경로에 저장
    cv2.imwrite(output_image_path, image)

filename = "201213_E_14_CCW_in_E_B_000_02237"

image_path = f"datasets/train/images/{filename}.png"
json_path = f"datasets/train/annotations/{filename}_BBOX.json"

for num in range(1, 4):
    output_path = f"/mnt/hdd_4A/choemj/2025winterlab/type_choemj/{num}.png"
    draw_bounding_boxes(image_path, json_path, output_path, num)
