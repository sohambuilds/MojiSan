import torch
import os
from dotenv import load_dotenv

load_dotenv()


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(os.getenv("APP_DEVICE", DEFAULT_DEVICE)) 

if DEVICE.type == 'cuda':
    TORCH_DTYPE = torch.float16
else:
    TORCH_DTYPE = torch.float32 

print(f"Using device: {DEVICE}")
print(f"Using dtype: {TORCH_DTYPE}")


# --- Model IDs ---
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_ENCODER_ID = "h94/IP-Adapter"
IMAGE_ENCODER_SUBFOLDER = "models/image_encoder"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"

# --- Adapter Weights ---
ADAPTER_WEIGHT_PLUS = "ip-adapter-plus_sdxl_vit-h.safetensors"
ADAPTER_WEIGHT_FACE = "ip-adapter-plus-face_sdxl_vit-h.safetensors"


DEFAULT_NEGATIVE_PROMPT = "multiple faces"
DEFAULT_STYLE_NEGATIVE_PROMPT = "photoreal"
DEFAULT_TEXT_NEGATIVE_PROMPT = "background, multiple images"
DEFAULT_STEPS = 50 # Balance speed/quality
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_STYLE_SCALE = 0.4
DEFAULT_FACE_SCALE = 0.7 #CHANGE


STYLE_LIBRARIES = {
    "ziggy": "static/styles/ziggy",
    "van_gogh": "static/styles/van_gogh",
    "pixel_art": "static/styles/pixel_art",
    "claymation": "static/styles/claymation",
    "sticker_art": "static/styles/sticker_art",
}
NUM_STYLE_IMAGES_PER_LIBRARY = 10