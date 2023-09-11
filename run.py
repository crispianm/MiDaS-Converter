# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_depth_estimate(filepath, transform, midas, device, color):
    # Transform input for midas
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    # Make a prediction
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()

    return output


# def split_video(video_filepath):
#     vidcap = cv2.VideoCapture(video_filepath)
#     count = 0
#     success = True
#     while success:
#         success,image = vidcap.read()
#         print('Read a new frame: ', success)
#         cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#         count += 1


def process_folder(
    input_folder_path, output_folder_path, transform, midas, device, color
):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for item in os.scandir(input_folder_path):
        if item.is_file():
            # Process file
            input_file_path = item.path
            output_file_path = os.path.join(
                output_folder_path, os.path.splitext(item.name)[0] + "_depth.png"
            )
            if os.path.exists(output_file_path):
                print(f"{output_file_path} already exists, skipping.")
            else:
                depth_estimate = get_depth_estimate(
                    input_file_path, transform, midas, device, color=color
                )
                print(f"Saving image {output_file_path}.")
                if color == True:
                    plt.imsave(output_file_path, depth_estimate, cmap="plasma")
                else:
                    plt.imsave(output_file_path, depth_estimate, cmap="binary")
        elif item.is_dir():
            # Recursively process subfolder
            subfolder_output_path = os.path.join(output_folder_path, item.name)
            process_folder(
                item.path, subfolder_output_path, transform, midas, device, color=color
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply MiDaS to a directory of images."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to folder containing images to apply MiDaS to. Ex: './data'",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output files to. Ex: './data_out'",
    )

    parser.add_argument(
        "-c",
        "--color",
        type=bool,
        default=False,
        choices=[True, False],
        help="Whether to output color images. (Default: False)",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="\nDevice to run the models. (Default: 'cuda')",
    )

    args = parser.parse_args()

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using ", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU instead")

    # Download the MiDaS model
    model_type = "DPT_BEiT_L_512"  # MiDaS v3.1 - Large     (highest accuracy, slowest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    # Define transforms for model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = (
        midas_transforms.beit512_transform
    )  # BEiT Transform appears better for DPT_BEiT_L_512 model, but could use DPT instead
    # transform = midas_transforms.dpt_transform

    process_folder(args.input, args.output, transform, midas, args.device, args.color)
