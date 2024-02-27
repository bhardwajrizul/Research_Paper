from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import warnings
warnings.filterwarnings("ignore")
print("Updated the code")

model = create_model("Unet_2020-10-30")
model.eval()
image = load_rgb("./static/cloth_web.jpg")

transform = albu.Compose([albu.Normalize(p=1)], p=1)

padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)

x = transform(image=padded_image)["image"]
x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

with torch.no_grad():
  prediction = model(x)[0][0]

mask = (prediction > 0).cpu().numpy().astype(np.uint8)
mask = unpad(mask, pads)

img = np.full((1024, 768, 3), 255)
seg_img = np.full((1024, 768), 0)

b = cv2.imread("./static/cloth_web.jpg")
b_img = mask * 255

if b.shape[1] <= 600 and b.shape[0] <= 500:
    scale_factor = min(1.2, 600 / b.shape[1], 500 / b.shape[0])
    b = cv2.resize(b, None, fx=scale_factor, fy=scale_factor)
    b_img = cv2.resize(b_img, (b.shape[1], b.shape[0]))  # Resize mask to match resized image

shape = b.shape  # Updated shape to match resized image

# Calculate start and end indices
start_x = int((1024 - shape[0]) / 2)
end_x = start_x + shape[0]
start_y = int((768 - shape[1]) / 2)
end_y = start_y + shape[1]

# Assign resized image and mask to the corresponding slice of img and seg_img
img[start_x:end_x, start_y:end_y] = b
seg_img[start_x:end_x, start_y:end_y] = b_img

cv2.imwrite("./HR-VITON-main/test/test/cloth/00001_00.jpg", img)
cv2.imwrite("./HR-VITON-main/test/test/cloth-mask/00001_00.jpg", seg_img)
