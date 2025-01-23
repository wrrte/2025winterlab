import os
import shutil

def move_images_to_root(image_folder):
    # 이미지 폴더의 하위 디렉토리들을 순회
    for root, dirs, files in os.walk(image_folder):
        if root == image_folder:
            # 최상위 디렉토리는 제외
            continue

        for file in files:
            file_path = os.path.join(root, file)
            # 파일이 이미지인지 확인 (확장자로 판단)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                try:
                    # 이미지 폴더 바로 아래로 이동
                    shutil.move(file_path, image_folder)
                    print(f"Moved: {file_path} -> {image_folder}")
                except Exception as e:
                    print(f"Error moving {file_path}: {e}")

    print("모든 이미지를 이동했습니다.")

# 사용 예시
image_folder = "train_bd/train/images"  # image 폴더의 경로
move_images_to_root(image_folder)