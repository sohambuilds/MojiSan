import os
import torch
from diffusers import FluxPipeline
from PIL import Image

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Correct FLUX.1 dev model ID from Black Forest Labs
MODEL_ID = "black-forest-labs/FLUX.1-dev"

NUM_IMAGES_PER_STYLE = 10
OUTPUT_ROOT = os.path.join("static", "styles")
SEED_BASE = 12345

# Updated style prompts to include single person's face with freely generated environment
STYLE_PROMPTS = {
    "ziggy": (
        "portrait of a single person's face, pop art, bold colors, comic-style, high contrast, "
        "dynamic background, no text, no watermark, simple composition"
    ),
    "van_gogh": (
        "portrait of a single person's face, in the style of Vincent van Gogh, thick brush strokes, "
        "swirling patterns, vivid colors, impressionist painting, any environment, "
        "no text, no watermark, simple composition"
    ),
    "pixel_art": (
        "portrait of a single person's face, pixel art, 32x32 grid, retro video game style, "
        "vibrant colors, dynamic background, no text, no watermark, simple composition"
    ),
    "claymation": (
        "portrait of a single person's face, claymation, stop-motion look, soft lighting, "
        "visible clay texture, any environment, no text, no watermark, simple composition"
    ),
    "sticker_art": (
        "portrait of a single person's face, flat vector sticker, thick outline, solid color fills, "
        "no gradients, dynamic background, no text, no watermark, simple composition"
    ),
}

# Updated negative prompt to enforce single person
NEGATIVE_PROMPT = (
    "blurry, text, watermark, border, multiple people, group, multiple objects, "
    "low quality, cropped, crowded, extra faces"
)

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Load pipeline
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    )

    # Move to device and enable optimizations
    pipe.to(DEVICE)
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()  # Save VRAM by offloading to CPU
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xformers not available, skipping.")
    pipe.enable_attention_slicing()  # Further memory optimization

    # Generate images per style
    for style, prompt in STYLE_PROMPTS.items():
        out_dir = os.path.join(OUTPUT_ROOT, style)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating {NUM_IMAGES_PER_STYLE} images for style '{style}'")

        for i in range(NUM_IMAGES_PER_STYLE):
            gen = torch.Generator(device="cpu").manual_seed(SEED_BASE + i)
            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=1024,  # FLUX.1 recommends 1024x1024 for high quality
                width=1024,
                num_inference_steps=50,  # Recommended for FLUX.1-dev
                guidance_scale=3.5,  # Suggested value for FLUX.1-dev
                max_sequence_length=512,  # FLUX.1 supports longer prompts
                generator=gen
            )
            img = result.images[0]
            img.save(os.path.join(out_dir, f"{i+1}.png"))

    print("Dataset generation complete.")

if __name__ == "__main__":
    main()