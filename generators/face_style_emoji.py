import torch
from PIL import Image
from typing import List
from utils.config import (
    DEVICE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_STEPS,
    DEFAULT_STYLE_SCALE,
    DEFAULT_FACE_SCALE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_STYLE_NEGATIVE_PROMPT
)

@torch.no_grad()
def generate_face_style_emoji(
    pipe,
    face_image: Image.Image,
    style_images: List[Image.Image],
    prompt: str,
    negative_prompt: str | None = None,  # Allow None from caller
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

    prompt = prompt if prompt else "emoji in a highly stylized artistic manner"
    final_negative_prompt = negative_prompt if negative_prompt is not None else DEFAULT_STYLE_NEGATIVE_PROMPT

    # Get the device directly from the pipeline
    device = pipe.device

    # Use "cpu" for generator to match diffusers' convention
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    adapter_scales = [face_scale] + [style_scale] * len(style_images)
    try:
        print(f"Set IP-Adapter scales: Style={style_scale}, Face={face_scale}")
    except ValueError as e:
        print(f"Error setting IP adapter scale (check number of loaded adapters): {e}")
        return None

    # Ensure all images are processed and moved to the pipeline's device
    style_tensors = [pipe.image_processor(image).to(device) for image in style_images]
    face_tensor = pipe.image_processor(face_image).to(device)

    ip_adapter_inputs = [face_tensor] + style_tensors

    print(f"Generating face+style emoji with prompt: '{prompt}'")
    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=final_negative_prompt,
            ip_adapter_image=ip_adapter_inputs,
            ip_adapter_scale=adapter_scales,
            num_inference_steps=num_inference_steps,
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
            generator=generator
        ).images[0]
        print("Face+style emoji generation complete.")
        return image
    except Exception as e:
        print(f"Error during face+style emoji generation: {e}")
        return None