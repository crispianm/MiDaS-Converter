# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

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


def get_depth_estimate(filename, model_type):

    # Input transformation pipeline
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    # DPT Transform appears better
    transform = midas_transforms.dpt_transform

    # if model_type == "MiDaS_small":
    #     transform = midas_transforms.small_transform
    # elif model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    #     transform = midas_transforms.dpt_transform
    # elif model_type == "DPT_BEiT_L_512":
    #     transform = midas_transforms.beit512_transform
    # elif (
    #     model_type == "DPT_SwinV2_L_384"
    #     or model_type == "DPT_SwinV2_B_384"
    #     or model_type == "DPT_Swin_L_384"
    # ):
    #     transform = midas_transforms.swin384_transform
    # elif model_type == "DPT_SwinV2_T_256":
    #     transform = midas_transforms.swin256_transform
    # elif model_type == "DPT_LeViT_224":
    #     transform = midas_transforms.levit_transform
    # else:
    # transform = midas_transforms.default_transform

    # Transform input for midas
    img = cv2.imread("./data/" + image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale (less good somehow)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    input_batch1 = transform(img).to(device)

    # Make a prediction
    with torch.no_grad():
        prediction = midas(input_batch1)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()

    return output


for image in os.listdir("./data/"):
    output = get_depth_estimate(image, model_type)
    directory = "./output/" + image.split(".")[0] + "_" + model_type + ".png"
    plt.axis("off")
    plt.imshow(output, cmap="plasma")
    plt.savefig(directory, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
