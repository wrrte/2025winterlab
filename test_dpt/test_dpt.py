import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add the parent folder to the Python path //dpt directory 문제해결
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

def load_images(image_folder):
    """Load all images from a folder."""
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images = [(path, cv2.imread(path)) for path in image_paths]
    return images

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

def evaluate_model(model, input_folder, ground_truth_folder, net_w, net_h, device):
    """Evaluate the DPT model using RMSE metric."""
    # Load input images and ground truth
    input_images = load_images(input_folder)
    ground_truth_images = load_images(ground_truth_folder)

    if len(input_images) != len(ground_truth_images):
        raise ValueError("Mismatch between input images and ground truth images")

    rmse_values = []

    for (input_path, input_image), (gt_path, gt_image) in zip(input_images, ground_truth_images):
        # Preprocess input image
        input_image_processed = preprocess_image(input_image, net_w, net_h)
        input_tensor = torch.from_numpy(input_image_processed).to(device).unsqueeze(0)

        # Predict depth map
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=gt_image.shape[:2], mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

        # Compute RMSE
        prediction = prediction.astype(np.float32)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Ensure unit consistency (convert to meters if needed)
        if np.max(gt_image) > 1000:  # Example condition for millimeters
            gt_image = gt_image / 1000.0
        if np.max(prediction) > 1000:  # Example condition for millimeters
            prediction = prediction / 1000.0

        # Ignore invalid depth values (assuming 0 indicates invalid depth)
        mask = gt_image > 0
        rmse = np.sqrt(mean_squared_error(gt_image[mask], prediction[mask]))
        rmse_values.append(rmse)

        print(f"Processed: {input_path}, RMSE: {rmse:.4f}")

    # Calculate average RMSE
    avg_rmse = np.mean(rmse_values)
    print(f"\nAverage RMSE: {avg_rmse:.4f}")
    return avg_rmse

if __name__ == "__main__":
    # Configuration
    model_paths = {
        "dpt_monodepth": "/home/seyeon/2025winterlab/test_dpt/weights/dpt_midas.pt",
        "dpt_kitti": "/home/seyeon/2025winterlab/test_dpt/weights/dpt_kitti.pt",
        "dpt_nyu": "/home/seyeon/2025winterlab/test_dpt/weights/dpt_nyu2.pt"
    }

    selected_model_type = "dpt_kitti"  #Select model to evaluate
    model_path = model_paths[selected_model_type]

    input_folder = "/home/seyeon/2025winterlab/test_dpt/dataset/image"  # Path to input images
    ground_truth_folder = "/home/seyeon/2025winterlab/test_dpt/dataset/groundtruth_depth"  # Path to ground truth images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nEvaluating model type: {selected_model_type}")
    model, net_w, net_h = load_model(model_path, selected_model_type, device)
    avg_rmse = evaluate_model(model, input_folder, ground_truth_folder, net_w, net_h, device)
    print(f"Finished evaluation for {selected_model_type}, Average RMSE: {avg_rmse:.4f}")
