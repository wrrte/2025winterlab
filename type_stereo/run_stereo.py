import cv2
from ultralytics.models import YOLO
import geopy.distance

# GPU 연결 코드

MODEL_PATH = "/home/seyeon/2025winterlab/type_stereo/test/best.pt"
model = YOLO(MODEL_PATH)

# Depth calculation constants
F = 2007.113 # focal length
B = 0.54 # baseline

current_gps = (37.5665, 126.9780) # Replace with real gps
heading = 180 # Replace with real heading

left_cam = cv2.VideoCapture(0)
right_cam = cv2.VideoCapture(1)

# # Camera resolution setting
# left_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

def calculate_gps_coordinates(current_gps, heading, angle, distance):
    actual_angle = (heading + angle) % 360
    origin = geopy.Point(current_gps[0], current_gps[1])
    destination = geopy.distance.distance(meters=distance).destination(origin, actual_angle)
    return destination.latitude, destination.longitude

def process_frame(left_frame, right_frame):
    left_results = model(left_frame)
    right_results = model(right_frame)

    left_boxes = left_results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
    right_boxes = right_results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
    
    # Match left/right bboxes
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
    return left_boxes, depths

def main(): 
    while True:
        # Capture frames
        ret_left, left_frame = left_cam.read()
        ret_right, right_frame = right_cam.read()

        if not ret_left or not ret_right:
            print("Error: Unable to capture frames from cameras")
            break

        left_boxes, depths = process_frame(left_frame, right_frame) # 메인함수 실행
        
        if left_boxes is None or depths is None:
            yield None, None, None
            continue  # Skip this frame and move to the next one

        gps_results = []
        for (x_min, y_min, x_max, y_max), depth in zip(left_boxes, depths):
            target_x = (x_min + x_max) / 2
            angle = (target_x - 320) / 320 * 35  # Assuming 640px width and 70-degree FOV
            gps_coordinates = calculate_gps_coordinates(current_gps, heading, angle, depth)
            gps_results.append((gps_coordinates, (x_min, y_min, x_max, y_max), depth))
      
                
        yield left_frame, left_boxes, gps_coordinates
    
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



print("Model run completed successfully.")

# Release the camera resources
left_cam.release()
right_cam.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    for frame, boxes, gps in main():
        if frame is None:
            print("No buildings detected in this frame.")
        else:
            print(f"Bounding Boxes: {boxes}")
            print(f"GPS Coordinates: {gps}")