import numpy as np
import cv2

with open("/mnt/hdd_4A/choemj/2025winterlab/2018-08-01-11-13-14_2018-08-01-11-51-30-941.pfm", "rb") as f:
        header = f.readline().decode('latin-1').rstrip()
        color = header == 'PF'
        
        dims = f.readline().decode('latin-1')
        width, height = map(int, dims.split())
        
        scale = float(f.readline().decode('latin-1').rstrip())
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'
        
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        
        prediction = np.reshape(data, shape)

min_val = np.min(prediction)
max_val = np.max(prediction)

if max_val - min_val > 0:  # 최대값과 최소값이 같으면 정규화 불가능
    scaled = (prediction - min_val) / (max_val - min_val)  # 0~1로 정규화
else:
    scaled = np.zeros_like(prediction)  # 모든 값이 같다면 0으로 채움

# 2. 0~255로 변환 후 uint8 타입으로 변경
scaled = (scaled * 255).astype(np.uint8)

# 3. PNG로 저장
cv2.imwrite("result.png", scaled)