import cv2
import numpy as np
import argparse
from ultralytics.models import YOLO
import glob
import os

# Depth calculation constants
F = 100 # focal length
B = 100 # baseline

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--model_type", default = "m", help = "YOLO model type to test")
args = parser.parse_args()

# Load model
model = YOLO(f"../weights/bd_trained/train5/weights/best.pt") #모델 경로 수정

#Load test dataset
left_images = sorted(glob.glob("test/test_left/*.png")) #left 이미지 경로
right_images = sorted(glob.glob("test/test_right/*.png")) #right 이미지 경로
depth_gt_images = sorted(glob.glob("test/test_depth/*.png")) #depth 이미지 경로
disparity_gt_images = sorted(glob.glob("test/test_disparity/*.png")) #disparity 이미지 경로

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
            cv2.rectangle(image, (int(x_min)), (int(y_min)), (int(x_max)), (int(y_max)), (255, 0, 0), 2)
    return image, valid_boxes

#Calculate RMSE
rmse_disparity = []
rmse_depth = []

# Image pair process
for i, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images)):
    left_image = cv2.imread(left_img_path)
    right_image = cv2.imread(right_img_path)
    depth_gt = np.load(depth_gt_images[i])
    disparity_gt = np.load(disparity_gt_images[i])

    left_results = model(left_img_path)
    right_results = model(right_img_path)

    left_image, left_boxes = draw_bounding_boxes(left_image, left_results)
    right_image, right_boxes = draw_bounding_boxes(right_image, right_results)

    # Save images with bboxes
    cv2.imwrite(f"{output_dir}/images/{os.path.basename(left_img_path)}", left_image)
    cv2.imwrite(f"{output_dir}/images/{os.path.basename(right_img_path)}", right_image)

    disparity = []
    depth = []

    