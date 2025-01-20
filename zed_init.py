import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import math
import sys
import pyzed.sl as sl
from datetime import datetime
import os
import time

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1200
init.camera_fps = 30
init.depth_mode = sl.DEPTH_MODE.NEURAL
init.coordinate_units = sl.UNIT.MILLIMETER
init.depth_stabilization = 100

runtime_parameters = sl.RuntimeParameters()
runtime_parameters.enable_fill_mode = True
runtime_parameters.confidence_threshold = 95

# Open the ZED camera
err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS:
    print(f"Failed to open ZED camera: {err}")
    exit(1)

# Prepare directories for saving images and coordinates
record_dir = "roadview/"
os.makedirs(record_dir, exist_ok=True)

# Prepare matrices for images and depth data
image = sl.Mat()
point_cloud = sl.Mat()

reference_point = 5000

def clear_previous_files(directory):
    """Delete all files in the specified directory."""
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def capture_image_and_point():
    global record_dir

    # Clear previous files before saving new data
    clear_previous_files(record_dir)

    # Capture a new image and point cloud
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image and point cloud
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # Convert the image to a numpy array
        frame = image.get_data()
        frame = cv2.resize(frame, (1920, 1200))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Flip the image both vertically and horizontally
        frame = cv2.flip(frame, -1)

        # Remove the bottom 100 pixels
        frame = frame[:1100, :]

        # Save the RGB image
        now = datetime.now()
        formatted_time = now.strftime("%d_%m_%Y_%H_%M_%S_%f")
        filename_str = "image_" + formatted_time + ".jpg"
        cv2.imwrite(record_dir + "image/" + filename_str, frame)

        # Find a 5m point along the central x-coordinate line in the point cloud
        found_point = False
        height, width = frame.shape[:2]
        center_x = width // 2  # Central x-coordinate

        for y in range(height):  # Iterate over the y-axis
            _, (rel_x, rel_y, rel_z, _) = point_cloud.get_value(center_x, y)
            if math.isfinite(rel_x) and math.isfinite(rel_y) and math.isfinite(rel_z):
                depth_value = math.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
                if abs(depth_value - reference_point) < 500:  # Check if the depth is close to 5m
                    found_point = True
                    selected_x, selected_y = center_x, y
                    break

        # Save the coordinates if a 5m point was found
        if found_point:
            # Adjust y-coordinate to account for the 100 pixels cut off
            selected_y = min(selected_y, 1100)
            print(f"5미터 지점 발견: 좌표 ({selected_x}, {selected_y})")
            coordinates_file = os.path.join(record_dir + "reference_point/", filename_str.replace('.jpg', '.txt'))
            with open(coordinates_file, 'w') as file:
                file.write(f"5m point at ({selected_x}, {selected_y})\n")
        else:
            print("5미터 지점 찾기 실패")

    # Release resources
    zed.close()
    cv2.destroyAllWindows()

# Start the image and point cloud capture process once
capture_image_and_point()