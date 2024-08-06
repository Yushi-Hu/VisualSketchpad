from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from typing import List
from torchvision.ops import box_convert
import gradio as gr

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


def detection(image, text, box_threshold=0.35, text_threshold=0.25):
    
    image_source, image = load_image(image)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    ret_json = {
        "boxes": boxes,
        "logits": logits,
        "phrases": phrases
    }
    
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=range(1, len(boxes)+1), phrases=phrases)
    
    annotated_pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    
    return annotated_pil_image, ret_json


demo = gr.Interface(fn=detection, inputs=[gr.Image(type="filepath"),
                                          "text",
                                          gr.Number(value=0.35),
                                          gr.Number(value=0.25)],
                    outputs=[gr.Image(type="pil"), "json"]
                    )
                    
demo.launch(share=True, server_name="localhost", server_port=8081)