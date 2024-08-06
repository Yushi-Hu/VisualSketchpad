import json
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image as PILImage
import base64
from io import BytesIO

def image_to_base64(pil_img):
    # Create a BytesIO buffer to save the image
    buffered = BytesIO()
    # Save the image in the buffer using a format like PNG
    pil_img.save(buffered, format="PNG")
    # Get the byte data from the buffer
    img_byte = buffered.getvalue()
    # Encode the bytes to base64
    img_base64 = base64.b64encode(img_byte)
    # Decode the base64 bytes to string
    return img_base64.decode('utf-8')


def custom_encoder(obj):
    """Custom JSON encoder function that replaces Image objects with '<image>'.
       Delegates the encoding of other types to the default encoder."""
    if isinstance(obj, Image.Image):
        return image_to_base64(obj)
    # Let the default JSON encoder handle any other types
    return json.JSONEncoder().default(obj)
