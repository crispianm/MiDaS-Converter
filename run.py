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

    # Transform input for midas
    img = cv2.imread("./data/" + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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


def tree2list(directory: str) -> list:
    import os

    tree = []
    for i in os.scandir(directory):
        if i.is_dir():
            tree.append(["Folder", i.name, i.path])
            tree.extend(tree2list(i.path))
        else:
            tree.append(["File", i.name, i.path])
    return tree


tree_list = tree2list("./data/")
if not os.path.exists("./output/"):
    os.makedirs("./output/")
    print("Creating folder: ./output/")

for tree in tree_list:
    tree[2] = tree[2].split("/")
    tree[2][1] = "output"

    if tree[0] == "Folder":
        if not os.path.exists("/".join(tree[2])):
            os.makedirs("/".join(tree[2]))
            print("Creating folder: ", "/".join(tree[2]))

    elif tree[0] == "File":
        output = get_depth_estimate(tree[2][-1], model_type)

        tree[2][-1] = tree[2][-1].split(".")[0] + "_depth.png"

        new_filepath = "/".join(tree[2])

        plt.axis("off")
        plt.imshow(output, cmap="plasma")
        plt.savefig(
            new_filepath, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0
        )

# for image in os.listdir("./data/"):
#     output = get_depth_estimate(image, model_type)
#     new_directory = "./output/" + image.split(".")[0] + "_" + model_type + ".png"
#     plt.axis("off")
#     plt.imshow(output, cmap="plasma")
#     plt.savefig(new_directory, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
