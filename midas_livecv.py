# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_type = "MiDaS_small"

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

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# Hook into OpenCV
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Transform input for midas
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to(device)

    # Make a prediction
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()

    plt.imshow(output, cmap="plasma")
    cv2.imshow("CV2Frame", frame)
    plt.pause(0.00001)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()

plt.show()
