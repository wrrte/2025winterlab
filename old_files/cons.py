import numpy as np
import cv2

from run_dpt import run_dpt

def draw_cross(image, center, size=100, color=(0, 0, 255), thickness=5):
    """
    이미지에 특정 좌표를 중심으로 십자를 그립니다.
    
    :param image: 이미지를 나타내는 NumPy 배열 (예: cv2.imread()로 읽은 이미지)
    :param center: 십자의 중심 좌표 (x, y)
    :param size: 십자의 크기 (기본값: 20)
    :param color: 십자의 색 (기본값: 빨강 (BGR))
    :param thickness: 선의 두께 (기본값: 2)
    """
    x, y = center
    
    # 수평선 (좌에서 우로)
    cv2.line(image, (x - size, y), (x + size, y), color, thickness)
    
    # 수직선 (위에서 아래로)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness)
    
    return image

input_file = "roadview/image/*.png"

prediction = run_dpt(input_file)

prediction_binary = np.where(prediction == 0, 0, 255)

prediction_binary = cv2.merge([prediction_binary, prediction_binary, prediction_binary])

draw_cross(prediction_binary, (960, 152))

cv2.imwrite("dpt_type/dpt_output/result.png", prediction_binary)