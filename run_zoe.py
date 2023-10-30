# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_depth_estimate(filepath, transform, zoe, device, color=True):
    # Transform input for zoe
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    # Make a prediction
    with torch.no_grad():
        prediction = zoe(input_batch)
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
    input_folder_path, output_folder_path, transform, zoe, device, color=True
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
                    input_file_path, transform, zoe, device, color=color
                )
                print(f"Saving image {output_file_path}.")
                if color == True:
                    plt.imsave(
                        output_file_path[:-4] + "_color" + ".png",
                        depth_estimate,
                        cmap="plasma",
                    )
                else:
                    plt.imsave(output_file_path, depth_estimate, cmap="binary")
        elif item.is_dir():
            # Recursively process subfolder
            subfolder_output_path = os.path.join(output_folder_path, item.name)
            process_folder(
                item.path, subfolder_output_path, transform, zoe, device, color=color
            )


if __name__ == "__main__":
    # todo: add command line arguments for model type and input/output folders

    model_type = "DPT_BEiT_L_512"

    # Download the zoe model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using ", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU instead")
    zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    zoe.to(device)
    zoe.eval()

    # Input transformation pipeline
    zoe_transforms = torch.hub.load("intel-isl/zoe", "transforms")

    # BEiT Transform appears better, but you could use DPT instead
    # transform = zoe_transforms.dpt_transform
    transform = zoe_transforms.beit512_transform

    # Set to False if you want grayscale output
    color = False

    # Check that the correct number of arguments have been provided
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input_folder output_folder")
    else:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        process_folder(
            input_folder, output_folder, transform, zoe, device, color=color
        )


# python run.py "./data" "./data_out"
"C:/Users/crisp/Desktop/layered-neural-atlases/data/puck"
"C:/Users/crisp/Desktop/layered-neural-atlases/data/puck_depth"
