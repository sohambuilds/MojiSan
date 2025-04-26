import requests
from io import BytesIO
from PIL import Image
from rembg import remove as remove_bg_rembg 
import os
import glob



def download_and_load_image(url: str) -> Image.Image | None:
    """Downloads an image from a URL and returns it as a PIL Image."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None

def load_image_from_path(path: str) -> Image.Image | None:
    """Loads an image from a local file path."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"Error loading image from path {path}: {e}")
        return None

def remove_background(image: Image.Image) -> Image.Image | None:
    """Removes the background from a PIL image using rembg."""
    try:
        
        processed_image = remove_bg_rembg(image)
        return processed_image
    except Exception as e:
        print(f"Error during background removal: {e}")
        return None 
def load_style_images(style_dir: str, num_images: int = 10) -> list[Image.Image]:
    """Loads up to num_images from a directory."""
    images = []
    if not os.path.isdir(style_dir):
        print(f"Error: Style directory not found: {style_dir}")
        return images

   
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_files.extend(glob.glob(os.path.join(style_dir, ext)))

    image_files.sort() # Optional: ensure consistent order

    count = 0
    for file_path in image_files:
        if count >= num_images:
            break
        img = load_image_from_path(file_path)
        if img:
            images.append(img)
            count += 1
        else:
            print(f"Warning: Could not load style image {file_path}")

    if count < num_images:
         print(f"Warning: Loaded only {count}/{num_images} images from {style_dir}")

    return images