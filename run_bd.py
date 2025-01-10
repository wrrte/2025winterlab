import cv2
from ultralytics.models import YOLOv10
import glob
import os

# 학습된 모델 로드
model = YOLOv10("./weight/bd_origin.pt")

def draw_bounding_boxes(image_path, results, output_image_path, output_coords_path, visualize_image_path, confidence_threshold=0.3):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    bounding_boxes = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
    confidences = results[0].boxes.conf.cpu().numpy()     # 신뢰도
    class_ids = results[0].boxes.cls.cpu().numpy()        # 클래스 ID
    
    with open(output_coords_path, 'w') as f:
        for bbox, conf, cls_id in zip(bounding_boxes, confidences, class_ids):
            if conf >= confidence_threshold:  # confidence 하한 설정
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                
                # 바운딩 박스의 중심 좌표 계산
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                # 바운딩 박스 좌표와 중심 좌표 저장
                f.write(f"{cls_id} {conf} {x_min/width} {y_min/height} {x_max/width} {y_max/height}\n")
                
                # 중심 좌표 출력
                print(f"Class ID: {cls_id}, Confidence: {conf}, Center (x, y): ({x_center}, {y_center})")
    
    # 결과 이미지를 지정된 경로에 저장
    cv2.imwrite(output_image_path, image)
    # 바운딩 박스가 그려진 이미지를 visualize 폴더에 저장
    cv2.imwrite(visualize_image_path, image)

# roadview 폴더에서 이미지 파일 검색
image_files = glob.glob('roadview/*.png')  # 폴더 내의 PNG 이미지 검색
if not image_files:
    print("No image found in the roadview folder.")
    exit(1)

# 폴더에 있는 첫 번째 이미지 파일 사용
source_image_path = image_files[0]
print(f"Using image: {source_image_path}")

# 결과를 저장할 경로 설정
output_image_path = 'upgrade_output/new_result.png'
output_coords_path = 'upgradebb_output/new_result.txt'
visualize_image_path = 'visualize/new_result_visualized.png'  # visualize 폴더에 저장될 이미지 경로

# 추론 수행
results = model(source_image_path)

# 바운딩 박스를 그려서 이미지와 좌표를 저장 (confidence threshold=0.3 적용)
draw_bounding_boxes(source_image_path, results, output_image_path, output_coords_path, visualize_image_path, confidence_threshold=0.3)

#branch test