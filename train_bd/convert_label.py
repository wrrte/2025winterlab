import json
import glob
import os


def json2txt(source_json_path, output_txt_path):

    with open(source_json_path) as ff:
        data = json.load(ff)

    height = data["imgHeight"]
    width = data["imgWidth"]
    
    with open(output_txt_path, 'w') as f:
        for object in data["objects"]:
            if object["label"] == "building":

                x_min = min(coord[0] for coord in object["polygon"])
                y_min = min(coord[1] for coord in object["polygon"])
                x_max = max(coord[0] for coord in object["polygon"])
                y_max = max(coord[1] for coord in object["polygon"])
                
                # 바운딩 박스의 중심 좌표 계산
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                # 바운딩 박스 좌표와 중심 좌표 저장
                f.write(f"0 1 {x_min/width} {y_min/height} {x_max/width} {y_max/height}\n")
                
                # 중심 좌표 출력
                print(f"Center (x, y): ({x_center}, {y_center})")


# 폴더 내의 모든 json 이미지 검색
json_files = glob.glob('train_bd/train/labels/**/*.json')  
if not json_files:
    print("No images found in the folder.")
    exit(1)

# 이미지 파일 처리
for source_json_path in json_files:
    # 원본 이미지 파일명 추출 (확장자 포함)
    file_name = os.path.basename(source_json_path)  
    file_name_without_ext = os.path.splitext(file_name)[0]  # 확장자를 제외한 파일명
    
    print(f"Processing image: {source_json_path}")
    
    # 결과를 저장할 경로 설정
    output_txt_path = f'train_bd/train/labels/txt/{file_name_without_ext}.txt'
    
    # 바운딩 박스를 그려서 이미지와 좌표를 저장 (confidence threshold=0.3 적용)
    json2txt(source_json_path, output_txt_path)
    
print("Processing completed.")
