import torch
from PIL import Image
from utils.config import DEVICE, DEFAULT_TEXT_NEGATIVE_PROMPT, DEFAULT_STEPS, DEFAULT_GUIDANCE_SCALE

@torch.no_grad()
def generate_text_emoji(
    pipe,
    prompt: str,
    negative_prompt: str | None = None, 
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int | None = None
) -> Image.Image | None:
    """Generates an emoji based purely on text prompt."""

    if pipe is None:
        print("Error: Text-to-Emoji pipeline is not loaded.")
        return None

    # Use default negative prompt if caller provides None or empty string
    final_negative_prompt = negative_prompt if negative_prompt else DEFAULT_TEXT_NEGATIVE_PROMPT

    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    print(f"Generating text emoji with prompt: '{prompt}'")
    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=final_negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        print("Text emoji generation complete.")
        return image
    except Exception as e:
        print(f"Error during text emoji generation: {e}")
        # Consider logging the full error: import traceback; traceback.print_exc()
        return None