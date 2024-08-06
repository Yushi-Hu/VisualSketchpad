from gradio_client import Client, file
from PIL import Image
import tempfile

# Set of Marks
som_client = Client("http://localhost:8080")

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
        
        som_params = {
            "granularity": granularity,
            "alpha": alpha,
            "label_mode": "Number",
            "anno_mode": anno_mode
        }

        outputs = som_client.predict(file(image), granularity, alpha, "Number", anno_mode)

        original_image = Image.open(image)
        output_image = Image.open(outputs[0])
        
    return output_image


testing_image = Image.open("vs_sign.jpeg")

output_image = segment_and_mark(testing_image)

output_image.save("output_image.jpg")