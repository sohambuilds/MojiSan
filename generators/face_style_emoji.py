import torch
from PIL import Image
from typing import List
from utils.config import (
    DEVICE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_STEPS,
    DEFAULT_STYLE_SCALE,
    DEFAULT_FACE_SCALE
)

@torch.no_grad()
def generate_face_style_emoji(
    pipe,
    face_image: Image.Image,
    style_images: List[Image.Image],
    prompt: str,
    negative_prompt: str | None = None, # Allow None from caller
    style_scale: float = DEFAULT_STYLE_SCALE,
    face_scale: float = DEFAULT_FACE_SCALE,
    num_inference_steps: int = DEFAULT_STEPS,
    seed: int | None = None
) -> Image.Image | None:
    """Generates a personalized emoji using face image, style images, and text prompt."""

    if pipe is None:
        print("Error: Face+Style pipeline is not loaded.")
        return None
    if not face_image or not style_images:
        print("Error: Missing face image or style images.")
        return None

    final_negative_prompt = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT

    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    adapter_scales = [style_scale, face_scale]
    try:
        pipe.set_ip_adapter_scale(adapter_scales)
    except ValueError as e:
         print(f"Error setting IP adapter scale (check number of loaded adapters): {e}")
         return None
    print(f"Set IP-Adapter scales: Style={adapter_scales[0]}, Face={adapter_scales[1]}")

    ip_adapter_inputs = [style_images, face_image]

    print(f"Generating face+style emoji with prompt: '{prompt}'")
    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=final_negative_prompt,
            ip_adapter_image=ip_adapter_inputs,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]
        print("Face+style emoji generation complete.")
        return image
    except Exception as e:
        print(f"Error during face+style emoji generation: {e}")
        # import traceback; traceback.print_exc()
        return None