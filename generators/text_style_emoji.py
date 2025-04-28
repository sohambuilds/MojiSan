import torch
from PIL import Image
from typing import List
from utils.config import (
    DEVICE,
    DEFAULT_STYLE_NEGATIVE_PROMPT,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STYLE_SCALE
)

@torch.no_grad()
def generate_text_style_emoji(
    pipe,
    style_images: List[Image.Image],
    prompt: str,
    negative_prompt: str | None = None, # Allow None from caller
    style_scale: float = DEFAULT_STYLE_SCALE,
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None
) -> Image.Image | None:
    """Generates an emoji using text prompt and style images."""

    if pipe is None:
        print("Error: Text+Style pipeline is not loaded.")
        return None
    
    pipe = pipe.to(DEVICE)
    if not style_images:
        print("Error: Missing style images.")
        return None

    final_negative_prompt = negative_prompt if negative_prompt else DEFAULT_STYLE_NEGATIVE_PROMPT

    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    try:
        pipe.set_ip_adapter_scale(style_scale)
    except ValueError as e:
         print(f"Error setting IP adapter scale (check number of loaded adapters): {e}")
         return None
    print(f"Set IP-Adapter style scale: {style_scale}")

    ip_adapter_inputs = [style_images]

    print(f"Generating text+style emoji with prompt: '{prompt}'")
    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=final_negative_prompt,
            ip_adapter_image=ip_adapter_inputs,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        print("Text+style emoji generation complete.")
        return image
    except Exception as e:
        print(f"Error during text+style emoji generation: {e}")
        # import traceback; traceback.print_exc()
        return None