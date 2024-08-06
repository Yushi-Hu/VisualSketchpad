from gradio_client import Client, file
from multimodal_conversable_agent import MultimodalConversableAgent
from PIL import Image, ImageDraw
import numpy as np
import tempfile
import time, os
import cv2, json, os, sys, time, random
import numpy as np
from PIL import Image
from matplotlib import colormaps
from matplotlib.colors import Normalize

# Load all vision experts
from config import SOM_ADDRESS, GROUNDING_DINO_ADDRESS, DEPTH_ANYTHING_ADDRESS

# Set of Marks
som_client = Client(SOM_ADDRESS)

# Grounding DINO
gd_client = Client(GROUNDING_DINO_ADDRESS)

# DepthAnything
da_client = Client(DEPTH_ANYTHING_ADDRESS)



class AnnotatedImage:
    # A class to represent an annotated image. It contains the annotated image and the original image.
    
    def __init__(self, annotated_image: Image.Image, original_image: Image.Image=None):
        self.annotated_image = annotated_image
        self.original_image = original_image



def segment_and_mark(image, granularity:float = 1.8, alpha:float = 0.1, anno_mode:list = ['Mask', 'Mark']):
    """Use a segmentation model to segment the image, and add colorful masks on the segmented objects. Each segment is also labeled with a number.
    The annotated image is returned along with the bounding boxes of the segmented objects.
    This tool may help you to better reason about the relationship between objects, which can be useful for spatial reasoning etc.

    Args:
        image (PIL.Image.Image): the input image
        granularity (float, optional): The granlarity of the segmentation. Rranges from 0 to 2.5. The higher the more fine-grained. Defaults to 1.8.
        alpha (float, optional): The alpha of the added colorful masks. Defaults to 0.1.
        anno_mode (list, optional): What annotation is added on the input image. Mask is the colorful masks. And mark is the number labels. Defaults to ['Mask', 'Mark'].

    Returns:
        output_image (AnnotatedImage): the original image annotated with colorful masks and number labels. Each mask is labeled with a number. The number label starts at 1.
        bboxes (List): listthe bounding boxes of the masks.The order of the boxes is the same as the order of the number labels.
        
    Example:
        User request: I want to find a seat close to windows, where should I sit?
        Code:
        ```python
        image = Image.open("sample_img.jpg")
        output_image, bboxes = segment_and_mark(image)
        display(output_image.annotated_image)
        ```
        Model reply: You can sit on the chair numbered as 5, which is close to the window.
        User: Give me the bounding box of that chair.
        Code:
        ```python
        print(bboxes[4]) # [0.24, 0.21, 0.3, 0.4]
        ```
        Model reply: The bounding box of the chair numbered as 5 is [0.24, 0.21, 0.3, 0.4].
    """
    
    
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        image.save(tmp_file.name, 'JPEG')
        image = tmp_file.name

        outputs = som_client.predict(file(image), granularity, alpha, "Number", anno_mode)

        original_image = Image.open(image)
        output_image = Image.open(outputs[0])
        
        output_image = AnnotatedImage(output_image, original_image)
        
        w,h = output_image.annotated_image.size
                
        masks = outputs[1]
        
        bboxes = []
        
        for mask in masks:
            bbox = mask['bbox']
            bboxes.append((bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h))
        
    return output_image, bboxes


def detection(image, objects, box_threshold:float = 0.35, text_threshold:float = 0.25):
    """Object detection using Grounding DINO model. It returns the annotated image and the bounding boxes of the detected objects.
    The text can be simple noun, or simple phrase (e.g., 'bus', 'red car'). Cannot be too hard or the model will break.
    The detector is not perfect, it may wrongly detect objects or miss some objects.
    Also, notice that the bounding box label might be out of the image boundary.
    You should use the output as a reference, not as a ground truth.

    Args:
        image (PIL.Image.Image): the input image
        objects (List[str]): a list of objects to detect. Each object should be a simple noun or a simple phrase.

    Returns:
        output_image (AnnotatedImage): the original image, annotated with bounding boxes. Each box is labeled with the detected object, and an index.
        processed boxes (List): listthe bounding boxes of the detected objects
    
    Example:
        image = Image.open("sample_img.jpg")
        output_image, boxes = detection(image, ["bus"])
        display(output_image.annotated_image)
        print(boxes) # [[0.24, 0.21, 0.3, 0.4], [0.6, 0.3, 0.2, 0.3]]
    """
    
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        image.save(tmp_file.name, 'JPEG')
        image = tmp_file.name
    
        outputs = gd_client.predict(file(image), ', '.join(objects), box_threshold, text_threshold)
        
        # process images
        original_image = Image.open(image)
        output_image = Image.open(outputs[0])
        output_image = AnnotatedImage(output_image, original_image)
        
        # process boxes
        boxes = outputs[1]['boxes']
        processed_boxes = []
        
        for box in boxes:
            processed_boxes.append((box[0]-box[2]/2, box[1] - box[3]/2, box[2], box[3]))
        
    return output_image, processed_boxes



def depth(image):
    """Depth estimation using DepthAnything model. It returns the depth map of the input image. 
    A colormap is used to represent the depth. It uses Inferno colormap. The closer the object, the warmer the color.
    This tool may help you to better reason about the spatial relationship, like which object is closer to the camera.
    Depth can also help you to understand the 3D structure of the scene, which can be useful for analyzing the motion between frames, etc.

    Args:
        image (PIL.Image.Image): the input image

    Returns:
        output_image (PIL.Image.Image): the depth map of the input image
        
    Example:
        image = Image.open("sample_img.jpg")S
        output_image = depth(image)
        display(output_image)
    """
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        image.save(tmp_file.name, 'JPEG')
        image = tmp_file.name
        outputs = da_client.predict(file(image))
    output_image = Image.open(outputs)
    
    return output_image


def crop_image(image, x:float, y:float, width:float, height:float):
    """Crop the image based on the normalized coordinates.
    Return the cropped image.
    This has the effect of zooming in on the image crop.

    Args:
        image (PIL.Image.Image): the input image
        x (float): the horizontal coordinate of the upper-left corner of the box
        y (float): the vertical coordinate of that corner
        width (float): the box width
        height (float): the box height

    Returns:
        cropped_img (PIL.Image.Image): the cropped image
        
    Example:
        image = Image.open("sample_img.jpg")
        cropped_img = crop_image(image, 0.2, 0.3, 0.5, 0.4)
        display(cropped_img)
    """
    
    # get height and width of image
    w, h = image.size
    
    # limit the range of x and y
    x = min(max(0, x), 1)
    y = min(max(0, y), 1)
    x2 = min(max(0, x+width), 1)
    y2 = min(max(0, y+height), 1)
    
    cropped_img = image.crop((x*w, y*h, x2*w, y2*h))
    return cropped_img


def zoom_in_image_by_bbox(image, box, padding=0.05):
    """A simple wrapper function to crop the image based on the bounding box.
    The zoom factor cannot be too small. Minimum is 0.1

    Args:
        image (PIL.Image.Image): the input image
        box (List[float]): the bounding box in the format of [x, y, w, h]
        padding (float, optional): The padding for the image crop, outside of the bounding box. Defaults to 0.05.

    Returns:
        cropped_img (PIL.Image.Image): the cropped image
        
    Example:
        image = Image.open("sample_img.jpg")
        annotated_img, boxes = detection(image, "bus")
        cropped_img = zoom_in_image_by_bbox(image, boxes[0], padding=0.1)
        display(cropped_img)
    """
    assert padding >= 0.05, "The padding should be at least 0.05"
    x, y, w, h = box
    x, y, w, h = x-padding, y-padding, w+2*padding, h+2*padding
    return crop_image(image, x, y, w, h)
        

def sliding_window_detection(image: Image.Image, objects):
    """Deal with the case when the user query is asking about objects that are not seen by the model.
    In that case, the most common reason is that the object is too small such that both the vision-language model and the object detection model fail to detect it.
    This function tries to detect the object by sliding window search.
    With the help of the detection model, it tries to detect the object in the zoomed-in patches.
    The function returns a list of annotated images that may contain at leas one of the objects, annotated with bounding boxes.
    It also returns a list of a list of bounding boxes of the detected objects.

    Args:
        image (PIL.Image.Image): the input image
        objects (List[str]): a list of objects to detect. Each object should be a simple noun or a simple phrase.
        
    Returns:
        possible_patches (List[AnnotatedImage]): a list of annotated zoomed-in images that may contain the object, annotated with bounding boxes.
        possible_boxes (List[List[List[Float]]]): For each image in possible_patches, a list of bounding boxes of the detected objects. 
            The coordinates are w.r.t. each zoomed-in image. The order of the boxes is the same as the order of the images in possible_patches.
            
    Example:
        image = Image.open("sample_img.jpg")
        possible_patches, possible_boxes = search_object_and_zoom(image, ["bird", "sign"])
        for i, patch in enumerate(possible_patches):
            print(f"Patch {i}:")
            display(patch.annotated_image)
        
        # print the bounding boxes of the detected objects in the first patch
        print(possible_boxes[0]) # [[0.24, 0.21, 0.3, 0.4], [0.6, 0.3, 0.2, 0.3]]
    """
    
    def check_if_box_margin(box, margin=0.005):
        x_margin = min(box[0], 1-box[0]-box[2])
        y_margin = min(box[1], 1-box[1]-box[3])
        return x_margin < margin or y_margin < margin
    
    # # first try to detect the object
    # annotated_img, detection_boxes = detection(image, text)
    
    # if len(detection_boxes) != 0:
    #     return [annotated_img], [detection_boxes]
    
    # if not detected, do sliding window search
    box_width = 1/3
    box_height = 1/3

    possible_patches = []
    possible_boxes = []
    
    for x in np.arange(0, 7/9, 2/9):
        for y in np.arange(0, 7/9, 2/9):
            cropped_img = crop_image(image, x, y, box_width, box_height)
            annotated_img, detection_boxes= detection(cropped_img, objects)
            
            # if one of the boxes is not too close to the edge, save it
            margin_flag = True
            for box in detection_boxes:
                if not check_if_box_margin(box):
                    margin_flag = False
                    break
            
            # if the object is detected and the box is not too close to the edge
            if len(detection_boxes) != 0 and not margin_flag:
                possible_patches.append(annotated_img)
                possible_boxes.append(detection_boxes)

    return possible_patches, possible_boxes


def overlay_images(background_img, overlay_img, alpha=0.3, bounding_box=[0, 0, 1, 1]):
    """
    Overlay an image onto another image with transparency.
    This is particularly useful visualizing heatmap while preserving some info from the original image.
    For example, you can overlay a segmented image on a heatmap to better understand the spatial relationship between objects.
    It will also help seeing the labels, circles on the original image that may not be visible on the heatmap.

    Args:
    background_img_pil (PIL.Image.Image): The background image in PIL format.
    overlay_img_pil (PIL.Image.Image): The image to overlay in PIL format.
    alpha (float): Transparency of the overlay image.
    bounding_box (List[float]): The bounding box of the overlay image. The format is [x, y, w, h]. The coordinates are normalized to the background image. Defaults to [0, 0, 1, 1].

    Returns:
    PIL.Image.Image: The resulting image after overlay, in PIL format.
    s
    Example:
        image = Image.open('original.jpg')
        depth_map = depth(image)
        overlayed_image = overlay_images(depth_map, image, alpha=0.3)
        display(overlayed_image)
    """
    # Calculate the actual pixel coordinates of the bounding box
    bg_width, bg_height = background_img.size
    x = int(bounding_box[0] * bg_width)
    y = int(bounding_box[1] * bg_height)
    w = int(bounding_box[2] * bg_width)
    h = int(bounding_box[3] * bg_height)

    # Resize overlay image to the bounding box size
    overlay_resized = overlay_img.resize((w, h), Image.Resampling.LANCZOS)

    # Adjust the overlay image's transparency
    overlay_with_alpha = overlay_resized.copy()
    overlay_with_alpha.putalpha(int(255 * alpha))  # Set the transparency level

    # Create a new image for the result and copy the background image to it
    new_img = Image.new('RGBA', background_img.size, (255, 255, 255, 255))
    new_img.paste(background_img, (0,0))

    # Paste the overlay image onto the new image with transparency
    new_img.paste(overlay_with_alpha, (x, y, x + w, y + h), overlay_with_alpha)

    return new_img.convert('RGB')  # Convert back to RGB if needed

