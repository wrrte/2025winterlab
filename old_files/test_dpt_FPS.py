import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# Add the parent folder to the Python path //dpt directory 문제해결
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2

# Set the random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def preprocess_image(image, net_w, net_h):
    """Preprocess an image for the DPT model."""
    transform = Compose([
        Resize(net_w, net_h, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        PrepareForNet()
    ])
    return transform({"image": image})["image"]

def load_model(model_path, model_type, device):
    """Load a specific DPT model with the appropriate parameters."""
    if model_type == "dpt_monodepth":
        net_w, net_h = 384, 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True
        )
    elif model_type == "dpt_kitti":
        net_w, net_h = 1216, 352
        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True
        )
    elif model_type == "dpt_nyu":
        net_w, net_h = 640, 480
        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True
        )
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")

    model.to(device)
    model.eval()
    return model, net_w, net_h

def calculate_fps(model, net_w, net_h, test_images_dir, device):
    """Calculate FPS for a given DPT model."""
    image_paths = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    inference_time = []
    fps = []

    for i in range(3):  # Repeat 10 times for averaging
        start_time = time.time()
        for image_path in image_paths:
            image = cv2.imread(image_path)
            input_image = preprocess_image(image, net_w, net_h)
            input_tensor = torch.from_numpy(input_image).to(device).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                _ = model(input_tensor)
        end_time = time.time()

        tot_time = end_time - start_time
        inference_time.append(tot_time)
        fps.append(len(image_paths) / tot_time)

        print(f"number of images : {len(image_paths)}")
        print(f"{i+1}th Inference Time: {inference_time[i]:.4f}s, {i+1}th FPS: {fps[i]:.4f}")

    # Calculate average FPS excluding outliers
    sorted_fps = sorted(fps)
    #trimmed_fps = sorted_fps[1:-1]
    #avg_fps = sum(trimmed_fps) / len(trimmed_fps)
    avg_fps = sum(sorted_fps) / len(sorted_fps)

    return avg_fps

if __name__ == "__main__":
    # Configuration
    model_paths = {
        "dpt_monodepth": "/home/seyeon/2025winterlab/test_dpt/weights/dpt_midas.pt",
        "dpt_kitti": "/home/seyeon/2025winterlab/test_dpt/weights/dpt_kitti.pt",
        "dpt_nyu": "/home/seyeon/2025winterlab/test_dpt/weights/dpt_nyu2.pt"
    }

    #######################SELECT MODEL############################
    selected_model_type = "dpt_nyu"
    model_path = model_paths[selected_model_type]
    ###############################################################

    test_images_dir = "/home/seyeon/2025winterlab/test_dpt/dataset/image"  # Path to input images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nCalculating FPS for model type: {selected_model_type}")
    model, net_w, net_h = load_model(model_path, selected_model_type, device)
    avg_fps = calculate_fps(model, net_w, net_h, test_images_dir, device)
    print(f"Finished calculating FPS for {selected_model_type}, Average FPS: {avg_fps:.4f}")

