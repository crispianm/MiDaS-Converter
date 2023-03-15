# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_type = "DPT_BEiT_L_512"

# Download the MiDaS
midas = torch.hub.load("intel-isl/MiDaS", model_type)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU instead")
midas.to(device)
midas.eval()

# Input transformation pipeline
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# DPT Transform appears better
transform = midas_transforms.dpt_transform


def get_depth_estimate(filepath, transform):

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


def process_folder(input_folder_path, output_folder_path, transform):
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
                depth_estimate = get_depth_estimate(input_file_path, transform)
                print(f"Saving image {output_file_path}.")
                plt.imsave(output_file_path, depth_estimate, cmap="plasma")
        elif item.is_dir():
            # Recursively process subfolder
            subfolder_output_path = os.path.join(output_folder_path, item.name)
            process_folder(item.path, subfolder_output_path, transform)


# python run.py "D:/LLVIP/data" "D:/LLVIP/depth"
# process_folder("./data", "./depth", transform)

if __name__ == "__main__":
    # Check that the correct number of arguments have been provided
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input_folder output_folder")
    else:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        process_folder(input_folder, output_folder, transform)
