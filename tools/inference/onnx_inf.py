"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import os
import cv2
import glob
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw


def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw(images, lines, scores):
    result_images = []
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        line = lines[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, l in enumerate(line):
            # Adjust bounding boxes according to the resizing and padding
            draw.line(list(l), fill="red", width=5)
            draw.text((l[0], l[1]), 
                text=f"{round(scrs[j].item(), 2)}",
                fill="blue")

        result_images.append(im)
    return result_images


def process_image(sess, im_pil):
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None]

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.538, 0.494, 0.453], std=[0.257, 0.263, 0.273]),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0)

    output = sess.run(
        output_names=None,
        input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
    )

    lines, scores = output

    result_images = draw([im_pil], lines, scores) #, [ratio], [(pad_w, pad_h)])
    result_images[0].save(f"{OUTPUT_NAME}.jpg")
    print(f"Image processing complete. Result saved as '{OUTPUT_NAME}.jpg'.")


def process_video(sess, video_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"{OUTPUT_NAME}.mp4", fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([w, h])[None]

        transforms = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
                T.Normalize(mean=[0.538, 0.494, 0.453], std=[0.257, 0.263, 0.273]),
            ]
        )
        im_data = transforms(frame_pil).unsqueeze(0)

        output = sess.run(
            output_names=None,
            input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
        )

        lines, scores = output

        # Draw detections on the original frame
        result_images = draw([frame_pil], lines, scores) #, [ratio], [(pad_w, pad_h)])
        frame_with_detections = result_images[0]

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved as '{OUTPUT_NAME}.mp4'.")

def process_file(sess, file_path):
    # Check if the input file is an image or a video
    try:
        # Try to open the input as an image
        im_pil = Image.open(file_path).convert("RGB")
        process_image(sess, im_pil)
    except IOError:
        # Not an image, process as video
        process_video(sess, file_path)

def main(args):
    # Global variable
    global OUTPUT_NAME , thrh

    """Main function."""
    # Load the ONNX model
    sess = ort.InferenceSession(args.onnx)
    print(f"Using device: {ort.get_device()}")

    input_path = args.input
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
            process_file(sess, file_path)
    else:
        # Process a file
        OUTPUT_NAME = 'onxx_results'
        process_file(sess, file_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input image or video file.")
    parser.add_argument("-t", "--thrh", type=float, required=False, default=None)
    args = parser.parse_args()
    main(args)
