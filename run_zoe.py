# Import dependencies
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_depth_estimate(filepath, zoe, device, color=True):

    img = Image.open(filepath).convert("RGB")

    # Make a depth prediction
    with torch.no_grad():
        prediction = zoe.infer_pil(img, output_type="tensor")
        output = prediction.cpu().numpy()
    return output



def process_folder(
    input_folder_path, output_folder_path, zoe, device, color=True
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
                    input_file_path, zoe, device, color=color
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
                item.path, subfolder_output_path, zoe, device, color=color
            )


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using ", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU instead")
    # zoe = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
        
    # ZoeD_NK

    conf = get_config("zoedepth_nk", "infer")
    zoe = build_model(conf)
    zoe.to(device)
    zoe.eval()

    # Set to False if you want grayscale output
    color = False

    # Check that the correct number of arguments have been provided
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input_folder output_folder")
    else:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        process_folder(
            input_folder, output_folder, zoe, device, color=color
        )


# python run_zoe.py "./data" "./data_out"
"C:/Users/crisp/Desktop/layered-neural-atlases/data/puck"
"C:/Users/crisp/Desktop/layered-neural-atlases/data/puck_depth"
