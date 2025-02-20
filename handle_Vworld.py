import requests
import cv2
import ast
import os
from geopy.distance import geodesic
from PIL import Image, ImageDraw, ImageFont
from dpt_type.GPS_dpt import GPS_dpt
from type_stereo.run_stereo import main as GPS_stereo

API_KEY = "3B8488FF-278C-3D24-992A-09B987D1CAB1"
API_URL = "https://api.vworld.kr/req/address"


predict_type = "dpt"
#predict_type = "stereo"
print(f"Running {predict_type} model ...\n")


current_gps = (37.5665, 126.9780) # Replace with real gps
heading = 180 # Replace with real heading

# DPT settings
ref_distance = 5.0
FOV = 72
model_path_dpt = "" # model dir

cap = cv2.VideoCapture(0)


while True:
    if predict_type == "dpt":
            image, detection_points, predicted_gps_points = GPS_dpt(
                 model_path_dpt, cap, current_gps, heading, ref_distance, FOV
                 )
    else:
        for image, detection_points, predicted_gps_points in GPS_stereo(current_gps, heading):
            break
    
    if image is None or not predicted_gps_points:
        print("No objects detected in this frame")
        continue

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Load font
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_size = 20
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # 출력된 내용을 문자열로 가져옴
    print("Fetching address from API ...\n")
    for idx, (gps_coordinate, bbox) in enumerate(zip(predicted_gps_points, detection_points)):
        latitude, longitude = gps_coordinate
        params = {
            "service": "address",
            "request": "getAddress",
            "format": "json",
            "crs": "epsg:4326",
            "point": f"{longitude},{latitude}",
            "type": "ROAD",
            "key": API_KEY
        }

        # Request address
        response = requests.post(API_URL, data=params)
        address = "Unknown Address"

        if response.status_code == 200:
            try:
                address = response.json()['response']['result'][0]['text']
                print(f"Detected Address: {address}")
            except Exception:
                print(f"API Error: {response.json()}")
        
        # Draw bounding box
        x_min, y_min, x_max, y_max = bbox
        draw.rectangle([x_min, y_min, x_max, y_max], outline = "red", width = 3)
        draw.text((x_min, y_min - 20), address, font = font, fill = "blue")

    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Detection Result", final_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
