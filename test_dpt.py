from ultralytics import YOLOv10

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 0, 1

import time
import torch

# Set the random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = YOLOv10('/home/seyeon/2025winterlab/runs/detect/train_m/weights/best.pt')
#model.train(data='yolov10.yaml', epochs=500, batch=16, imgsz=640)
test_images_dir = '/home/seyeon/2025winterlab/train_bd/train/images/'

inference_time = []
fps = []

for i in range (10):

    #fps 계산
    start_time = time.time()
    outputs = model(test_images_dir)
    end_time = time.time()
    tot_time = end_time - start_time

    inference_time.append(tot_time)
    num_images = len(os.listdir(test_images_dir))
    fps.append(num_images / tot_time)

for i in range(10):
    print(f"{i+1}th Inference Time: {inference_time[i]:.4f}s, {i+1}th FPS: {fps[i]:.4f}")

sorted_fps = sorted(fps)
trimmed_fps = sorted_fps[1:-1]
avg_fps = sum(sorted_fps) / len(sorted_fps)

print(f"Average FPS : {avg_fps:.4f}")




"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse

import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

import numpy as np

#from util.misc import visualize_attention


def run(input_path, output_path, model_path, model_type="dpt_monodepth", optimize=True, cuda_NUM=0):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
        model_type (str): type of the model to use [dpt_monodepth|dpt_kitti|dpt_nyu]
        optimize (bool): whether to optimize model for inference
        cuda_NUM (int): GPU device number
    """
    print("initialize")

    # select device (set to GPU cuda_NUM)
    device = torch.device(f"cuda:{cuda_NUM}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load network
    if model_type == "dpt_monodepth":
        net_w, net_h = 384, 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_kitti":
        net_w, net_h = 1216, 352
        model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_nyu":
        net_w, net_h = 640, 480
        model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    else:
        raise ValueError(
            f"model_type '{model_type}' not implemented. Use one of: [dpt_monodepth|dpt_kitti|dpt_nyu]"
        )

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue

        print(f"  processing {img_name} ({ind + 1}/{num_images})")
        img = util.io.read_image(img_name)

        # 디버깅 코드 추가: 입력 이미지 확인
        if img is None:
            raise ValueError(f"Input image at {img_name} could not be loaded!")
        print(f"Input image shape: {img.shape}, range: [{img.min()}, {img.max()}]")

        if model_type == "dpt_kitti" and args.kitti_crop:
            height, width, _ = img.shape
            top = height - 352
            left = (width - 1216) // 2
            img = img[top : top + 352, left : left + 1216, :]

        img_input = transform({"image": img})["image"]
        # 디버깅 코드 추가: 전처리된 입력 데이터 확인
        print(f"Preprocessed input tensor shape: {img_input.shape}, "
            f"Raw range: [{img_input.min()}, {img_input.max()}]")

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device.type == 'cuda':
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            # 디버깅 코드 추가: 모델 출력 확인
            print(f"Model output shape: {prediction.shape}, "
                f"range: [{prediction.min()}, {prediction.max()}]")
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            if model_type == "dpt_kitti":
                prediction *= 256

            if model_type == "dpt_nyu":
                prediction *= 1000.0

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        util.io.write_depth(f"{filename}_{model_type}", prediction, bits=1, absolute_depth=args.absolute_depth)

    print("finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="roadview/image/", help="folder with input images"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="dpt_output",
        help="folder for output images",
    )

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_monodepth",
        help="model type [dpt_monodepth|dpt_kitti|dpt_nyu]",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")

    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    default_models = {
        "dpt_monodepth": "weights/dpt_hybrid/dpt_monodepth.pt",
        "dpt_kitti": "weights/dpt_hybrid/dpt_kitti.pt",
        "dpt_nyu": "weights/dpt_hybrid/dpt_nyu.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )