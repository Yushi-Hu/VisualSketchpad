import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(DEVICE).eval()


def predict_depthmap(image):
    original_image = image.copy()

    h, w = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = model(image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    
    # colored_depth = Image.fromarray(cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB))
    corlored_depth = Image.fromarray(colored_depth)
    
    return colored_depth


demo = gr.Interface(fn=predict_depthmap, inputs=[gr.Image()],
                    outputs=[gr.Image(type="pil")]
                    )
                    
demo.launch(share=True, server_name="localhost", server_port=8082)



