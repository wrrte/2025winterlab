import cv2
import numpy as np
from ultralytics.models import YOLO
import glob
import os
from sklearn.metrics import mean_squared_error

# Depth calculation constants
F = 100 # focal length
B = 100 # baseline

# Select model
MODEL_PATH = "/home/seyeon/2025winterlab/type_seyeon/weights/bd_trained/train10/weights/best.pt"
model = YOLO(MODEL_PATH)

#Load test dataset
left_images = sorted(glob.glob("test/test_left/*.png"))
right_images = sorted(glob.glob("test/test_right/*.png"))
depth_gt_images = sorted(glob.glob("test/test_depth/*.png"))
disparity_gt_images = sorted(glob.glob("test/test_disparity/*.png"))

# Set output directory
output_dir = "test_output"
os.makedirs(output_dir, exist_ok = True)
os.makedirs(f"{output_dir}/images", exist_ok = True)
os.makedirs(f"{output_dir}/disparity", exist_ok = True)
os.makedirs(f"{output_dir}/depth", exist_ok = True)

# Draw bounding box - function
def draw_bounding_boxes(image, results, confidence_threshold = 0.3):
    bounding_boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    valid_boxes = []
    for bbox, conf, cls_id in zip(bounding_boxes, confidences, class_ids):
        if conf >= confidence_threshold:
            valid_boxes.append((cls_id, conf, bbox))
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
    return image, valid_boxes


# Image pair process
for i, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images)):
    left_image = cv2.imread(left_img_path)
    right_image = cv2.imread(right_img_path)
    depth_gt = cv2.imread(depth_gt_images[i], cv2.IMREAD_GRAYSCALE)
    disparity_gt = cv2.imread(disparity_gt_images[i], cv2.IMREAD_GRAYSCALE)


    left_results = model(left_img_path)
    right_results = model(right_img_path)

    left_image, left_boxes = draw_bounding_boxes(left_image, left_results)
    right_image, right_boxes = draw_bounding_boxes(right_image, right_results)

    # Save images with bboxes
    cv2.imwrite(f"{output_dir}/images/{os.path.basename(left_img_path)}", left_image)
    cv2.imwrite(f"{output_dir}/images/{os.path.basename(right_img_path)}", right_image)

    disparity_list = []
    depth_list = []

    for(cls_l, conf_l, bbox_l) in left_boxes:
        for(cls_r, conf_r, bbox_r) in right_boxes:
            if cls_l == cls_r:
                x_min_l, _, x_max_l, _ = bbox_l
                x_min_r, _, x_max_r, _ = bbox_r

                x_center_l = (x_min_l + x_max_l) / 2
                x_center_r = (x_min_r + x_max_r) / 2
                disparity = abs(x_center_l - x_center_r)
                depth = (F * B) / disparity if disparity != 0 else 0

                disparity_list.append(disparity)
                depth_list.append(depth)
    
    # Save as txt
    base_filename = os.path.splitext(os.path.basename(left_img_path))[0]
    with open(f"{output_dir}/disparity/{base_filename}.txt", "w") as disp_file:
        disp_file.write("\n".join(map(str, disparity_list)))

    with open(f"{output_dir}/depth/{base_filename}.txt", "w") as depth_file:
        depth_file.write("\n".join(map(str, depth_list)))

    # Compute RMSE
    rmse_disparity_list = []
    rmse_depth_list = []

    if len(disparity_list) > 0:
        disparity_pred = np.array(disparity_list)
        depth_pred = np.array(depth_list)

        disparity_gt_resized = disparity_gt[:len(disparity_pred)]
        depth_gt_resized = depth_gt[:len(depth_pred)]

        rmse_disparity = np.sqrt(mean_squared_error(disparity_gt_resized, disparity_pred))
        rmse_depth = np.sqrt(mean_squared_error(depth_gt_resized, depth_pred))

        rmse_disparity_list.append(rmse_disparity)
        rmse_depth_list.append(rmse_depth)
    
    # Print RMSE results
    print(f"Average RMSE for Disparity: {np.mean(rmse_disparity_list):.4f}")
    print(f"Average RMSE for Depth: {np.mean(rmse_depth_list):.4f}")