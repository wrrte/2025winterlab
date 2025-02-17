import cv2
from ultralytics.models import YOLO
import glob
import os
from sklearn.metrics import mean_squared_error
import numpy as np


MODEL_PATH = "/home/seyeon/2025winterlab/type_seyeon/weights/bd_trained/train10/weights/best.pt"
model = YOLO(MODEL_PATH)

# Depth calculation constants
F = 2007.113 # pixels
B = 0.54 # meters

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

 
left_imgs = sorted(glob.glob("/home/seyeon/2025winterlab/type_seyeon/test/test_left/*.png"))
right_imgs = sorted(glob.glob("/home/seyeon/2025winterlab/type_seyeon/test/test_right/*.png"))
depth_gt_imgs = sorted(glob.glob("/home/seyeon/2025winterlab/type_seyeon/test/test_depth/*.png"))
disparity_gt_imgs = sorted(glob.glob("/home/seyeon/2025winterlab/type_seyeon/test/test_disparity/*.png"))

if not left_imgs:
    print("No images found in the test_left folder.")
    exit(1)

if not depth_gt_imgs:
    print("No images found in the test_depth folder.")
    exit(1)

output_dir = "/home/seyeon/2025winterlab/type_seyeon/test/test_results"
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/coordinates", exist_ok=True)
os.makedirs(f"{output_dir}/disparity", exist_ok=True)
os.makedirs(f"{output_dir}/depth", exist_ok=True)

rmse_disparity_list = []
rmse_depth_list = []

# 이미지 파일 처리
for left_img_path, right_img_path, depth_gt_path, disparity_gt_path in zip(left_imgs, right_imgs, depth_gt_imgs, disparity_gt_imgs):
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    depth_img = cv2.imread(depth_gt_path, cv2.IMREAD_GRAYSCALE)
    disparity_img = cv2.imread(disparity_gt_path, cv2.IMREAD_GRAYSCALE)

    left_results = model(left_img_path)
    right_results = model(right_img_path)

    base_filename = os.path.basename(left_img_path)

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
    

    # Extract ground truth disparity values at bounding box center locations
    disparity_gt_values = []
    depth_gt_values = []

    for bbox in left_boxes:
        x_center = int((bbox[0] + bbox[2]) / 2)  # Bounding box center x
        y_center = int((bbox[1] + bbox[3]) / 2)  # Bounding box center y
        
        # Ensure the center is within bounds of the image
        if 0 <= y_center < disparity_img.shape[0] and 0 <= x_center < disparity_img.shape[1]:
            disparity_gt_values.append(disparity_img[y_center, x_center])
            depth_gt_values.append(depth_img[y_center, x_center])

    # Compute RMSE only if we have valid matches
    if len(disparity_gt_values) > 0 and len(disparities) > 0:
        rmse_disparity = np.sqrt(mean_squared_error(disparity_gt_values, disparities))
        rmse_depth = np.sqrt(mean_squared_error(depth_gt_values, depths))
        rmse_disparity_list.append(rmse_disparity)
        rmse_depth_list.append(rmse_depth)



print(f"Average RMSE for Disparity: {np.mean(rmse_disparity_list):.4f}")
print(f"Average RMSE for Depth: {np.mean(rmse_depth_list):.4f}")