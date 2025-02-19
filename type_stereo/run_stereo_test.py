import cv2
from ultralytics.models import YOLO
import glob
import os
import numpy as np

# GPU 연결 코드

MODEL_PATH = "/home/seyeon/2025winterlab/type_stereo/test/best.pt"
model = YOLO(MODEL_PATH)

# Depth calculation constants
F = 2007.113 # focal length
B = 0.54 # baseline

def draw_bounding_boxes(image_path, results, output_coords_path, confidence_threshold=0.3):
    image = cv2.imread(image_path)
    
    bounding_boxes = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
    confidences = results[0].boxes.conf.cpu().numpy()     # 신뢰도
    valid_boxes = []
    
    if output_coords_path is not None:
        with open(output_coords_path, 'w') as f:
            for bbox, conf in zip(bounding_boxes, confidences):
                if conf >= confidence_threshold:
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
                    valid_boxes.append([x_min, y_min, x_max, y_max])
                    f.write(" ".join(map(str, [x_min, y_min, x_max, y_max])) + "\n")

    else:
        for bbox, conf in zip(bounding_boxes, confidences):
            if conf >= confidence_threshold:
                x_min, y_min, x_max, y_max = bbox
                valid_boxes.append([x_min, y_min, x_max, y_max])

    return image, valid_boxes

 
left_imgs = sorted(glob.glob("/home/seyeon/2025winterlab/type_stereo/test/test_left/*.png")) # 왼쪽 카메라 dir
right_imgs = sorted(glob.glob("/home/seyeon/2025winterlab/type_stereo/test/test_right/*.png")) # 오른쪽 카메라 dir

if not left_imgs or len(left_imgs) != len(right_imgs):
    print("Missing images or missmatch in the input images.")
    exit(1)

output_dir = "/home/seyeon/2025winterlab/type_stereo/test/test_results" # 경로
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/coordinates", exist_ok=True)
os.makedirs(f"{output_dir}/disparity", exist_ok=True)
os.makedirs(f"{output_dir}/depth", exist_ok=True)

# 이미지 파일 처리
for left_img_path, right_img_path in zip(left_imgs, right_imgs):
    base_filename = os.path.basename(left_img_path)

    # batch_results = model([left_img_path, right_img_path], device = "cuda")
    # left_results, right_results = batch_results

    left_results = model(left_img_path)
    right_results = model(right_img_path)

    left_img, left_boxes = draw_bounding_boxes(left_img_path, left_results, f"{output_dir}/coordinates/{base_filename}.txt")
    cv2.imwrite(f"{output_dir}/images/{base_filename}", left_img)

    # Get bounding boxes for RIGHT image (used only for matching)
    _, right_boxes = draw_bounding_boxes(right_img_path, right_results, None)  # Not saving right coords

    # match buildings in pairs
    disparities = []
    depths = []

    for bbox_l in left_boxes:
        x_center_l = (bbox_l[0] + bbox_l[2]) / 2
        best_match = None
        min_distance = float('inf')

        for bbox_r in right_boxes:
            x_center_r = (bbox_r[0] + bbox_r[2]) / 2
            distance = abs(x_center_l - x_center_r)
            if distance < min_distance:
                min_distance = distance
                best_match = bbox_r
        
        if best_match:
            disparity = abs(x_center_l - ((best_match[0] + best_match[2]) / 2) )
            depth = (F * B) / max(disparity, 1e-6)
            disparities.append(disparity)
            depths.append(depth)

    # Save disparity values
    with open(f"{output_dir}/disparity/{base_filename}.txt", "w") as disp_file:
        disp_file.write("\n".join(map(str, disparities)))

    # Save depth values
    with open(f"{output_dir}/depth/{base_filename}.txt", "w") as depth_file:
        depth_file.write("\n".join(map(str, depths)))

    print("Model run completed successfully.")