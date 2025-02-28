from PIL import Image
import numpy as np

def load_image_and_get_dimensions(image_path):
    """
    Load an image, convert it to an RGB numpy array, and return its pixels, height, and width.

    :param image_path: Path to the image file.
    :return: A tuple containing the pixel array, height, and width of the image.
    """
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB mode
    pixels = np.array(image)  # Convert to numpy array
    height, width = pixels.shape[:2]  # Get image dimensions
    return pixels, height, width  # Corrected return order

def create_image_from_pixels(pixels, height, width, save_path=None):
    """
    Create an image from the given pixel array and optionally save it.

    :param pixels: 2D numpy array of pixel values.
    :param height: Image height.
    :param width: Image width.
    :param save_path: If provided, the image will be saved to this path.
    :return: PIL Image object.
    """
    pixels = pixels.reshape(height, width, 3)  # Corrected shape
    image = Image.fromarray(pixels.astype("uint8"))  # Ensure correct format
    
    if save_path:
        image.save(save_path)  # Save the image if a path is given
    return image
