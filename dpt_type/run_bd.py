import cv2
from ultralytics.models import YOLO
import glob
import os
import argparse
import torch

def run_bd(image, model, output_image_path = 'dpt_type/bd_output/image/result.png', output_coords_path='dpt_type/bd_output/coordinate/result.txt', visualize_image_path='dpt_type/bd_output/visualize/result.png', confidence_threshold=0.3):

    

    results = model(image, imgsz=640)

    height, width = image.shape[:2]
    
    bounding_boxes = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
    confidences = results[0].boxes.conf.cpu().numpy()     # 신뢰도
    class_ids = results[0].boxes.cls.cpu().numpy()        # 클래스 ID
    
    detection_points = []

    idx = 0
    with open(output_coords_path, 'w') as f:
        for idx, (bbox, conf, cls_id) in enumerate(zip(bounding_boxes, confidences, class_ids)):
            if conf >= confidence_threshold:  # confidence 하한 설정
                idx += 1
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                cv2.putText(
    image, 
    str(idx),  # 표시할 숫자
    (int(x_min) + 3, int(y_min) - 5),  # 위치 (사각형 왼쪽 위 근처)
    cv2.FONT_HERSHEY_SIMPLEX,  # 폰트
    1,  # 글자 크기 (작게)
    (255, 0, 0),  # 색상 (BGR)
    1,  # 두께
    cv2.LINE_AA  # 부드러운 글자 스타일
)
                # 바운딩 박스의 중심 좌표 계산
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                # 바운딩 박스 좌표와 중심 좌표 저장
                f.write(f"{cls_id} {conf} {x_min/width} {y_min/height} {x_max/width} {y_max/height}\n")
                
                # 1층 부근 좌표 출력
                #detection_points.append((x_center, (y_max+y_center)/2))

                #중심 좌표 출력
                detection_points.append((x_center, y_center))

    
    # 결과 이미지를 지정된 경로에 저장
    cv2.imwrite(output_image_path, image)
    # 바운딩 박스가 그려진 이미지를 visualize 폴더에 저장
    cv2.imwrite(visualize_image_path, image)

    return detection_points

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"]="3" # 0, 1

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--model_type", default="m", help="yolo model type you want to run")
    args = parser.parse_args()
    #model = YOLO(f"/mnt/hdd_4A/choemj/2025winterlab/weights/trained/train_{args.model_type}/weights/best.pt")
    #model = YOLO("/mnt/hdd_4A/choemj/2025winterlab/type_30000/weight/train3/weights/best.pt")
    model = YOLO("/mnt/hdd_4A/choemj/2025winterlab/type_xloss/weights/train/weights/best.pt")


    # roadview/image 폴더 내의 모든 PNG 이미지 검색
    image_files = glob.glob('roadview/*.png')  
    if not image_files:
        print("No images found in the roadview folder.")
        exit(1)

    # 이미지 파일 처리
    for image_path in image_files:
        # 원본 이미지 파일명 추출 (확장자 포함)
        file_name = os.path.basename(image_path)  
        file_name_without_ext = os.path.splitext(file_name)[0]  # 확장자를 제외한 파일명
        
        #print(f"Processing image: {source_image_path}")
        
        # 결과를 저장할 경로 설정
        output_image_path = f'dpt_type/bd_output/image/{file_name_without_ext}.png'
        output_coords_path = f'dpt_type/bd_output/coordinate/{file_name_without_ext}.txt'
        visualize_image_path = f'dpt_type/bd_output/visualize/{file_name_without_ext}-visualized.png'
        
        # 추론 수행



        image = cv2.imread(image_path)
        
        # 바운딩 박스를 그려서 이미지와 좌표를 저장 (confidence threshold=0.3 적용)
        print(run_bd(image, model, output_image_path, output_coords_path, visualize_image_path))
        
    print("Processing completed for all images.")
