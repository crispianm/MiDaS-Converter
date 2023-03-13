# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_type = "DPT_Large"

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

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Transform input for midas
    img = cv2.imread("./data/" + image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch1 = transform(img).to(device)

    # Make a prediction 1
    with torch.no_grad():
        prediction1 = midas(input_batch1)
        prediction1 = torch.nn.functional.interpolate(
            prediction1.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction1.cpu().numpy()

    return output


for image in os.listdir("./data/"):
    output = get_depth_estimate(image, model_type)
    plt.imshow(output, cmap="plasma")
    plt.savefig("./output/" + image.split(".")[0] + ".png")
