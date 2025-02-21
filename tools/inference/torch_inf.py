"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import os
import sys
import glob

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from util.slconfig import SLConfig


def draw(images, lines, scores):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        line = lines[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, l in enumerate(line):
            draw.line(list(l), fill="red", width=5)
            draw.text(
                (l[0], l[1]),
                text=f"{round(scrs[j].item(), 2)}",
                fill="blue",
            )

    return images 


def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.538, 0.494, 0.453], std=[0.257, 0.263, 0.273]),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    lines, scores = output

    result_images = draw([im_pil], lines, scores)
    result_images[0].save(f"{OUTPUT_NAME}.jpg")
    print(f"Image processing complete. Result saved as '{OUTPUT_NAME}.jpg'.")


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{OUTPUT_NAME}.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.538, 0.494, 0.453], std=[0.257, 0.263, 0.273]),
        ]
    )

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        lines, scores = output

        # Draw detections on the frame
        result_images = draw([frame_pil], lines, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(result_images[0]), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")

def process_file(model, device, file_path):
    # Check if the input file is an image or a vide
    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(model, device, file_path)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path)
        print("Video processing complete.")

def create(args, classname):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    class_module = getattr(args, classname)
    assert class_module in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(class_module)
    return build_func(args)

def main(args):
    # Global variable
    global OUTPUT_NAME, thrh 

    """Main function"""
    cfg = SLConfig.fromfile(args.config)

    if 'HGNetv2' in cfg.backbone:
        cfg.pretrained = False

    cfg.multiscale = None

    # build model
    model, postprocessor = create(cfg, 'modelname')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
        model.load_state_dict(state)
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    thrh = 0.4 if args.thrh is None else args.thrh

    # Check if the input argumnet is a file or a folder
    file_path = args.input
    if os.path.isdir(file_path):
        # Process a folder
        folder_dir = args.input
        output_dir = f"{folder_dir}/output"
        os.makedirs(output_dir, exist_ok=True)
        paths = list(glob.iglob(f"{folder_dir}/*.*"))
        for file_path in paths:
            OUTPUT_NAME = file_path.replace(f'{folder_dir}/', f'{output_dir}/').split('.')[0]
            process_file(model, device, file_path)
    else:
        # Process a file
        OUTPUT_NAME = 'torch_results'
        process_file(model, device, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-t", "--thrh", type=float, required=False, default=None)
    args = parser.parse_args()
    main(args)
