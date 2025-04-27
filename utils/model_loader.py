import torch
from diffusers import (
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
    DDIMScheduler
)
from transformers import CLIPVisionModelWithProjection
from .config import (
    DEVICE, TORCH_DTYPE, BASE_MODEL_ID,
    IMAGE_ENCODER_ID, IMAGE_ENCODER_SUBFOLDER,
    IP_ADAPTER_REPO, IP_ADAPTER_SUBFOLDER,
    ADAPTER_WEIGHT_PLUS, ADAPTER_WEIGHT_FACE
)
import gc


loaded_models = {
    "text_pipe": None,
    "face_style_pipe": None,
    "text_style_pipe": None,
    "image_encoder": None,
}

def move_pipeline_to_device(pipe, device):
    if pipe is not None:
        pipe.to(device)
        if hasattr(pipe, "components") and "image_encoder" in pipe.components:
            pipe.components["image_encoder"].to(device)

def ensure_only_one_on_gpu(target_key):
    """Moves all pipelines to CPU except the one specified by target_key (which is moved to GPU)."""
    for key, pipe in loaded_models.items():
        if pipe is not None:
            if key == target_key:
                move_pipeline_to_device(pipe, DEVICE)
            else:
                move_pipeline_to_device(pipe, "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_image_encoder():
    """Loads the CLIP Vision Image Encoder."""
    if loaded_models["image_encoder"] is None:
        print("Loading Image Encoder...")
        try:
            encoder = CLIPVisionModelWithProjection.from_pretrained(
                IMAGE_ENCODER_ID,
                subfolder=IMAGE_ENCODER_SUBFOLDER,
                torch_dtype=TORCH_DTYPE,
            ).to(DEVICE)
            loaded_models["image_encoder"] = encoder
            print("Image Encoder loaded.")
        except Exception as e:
            print(f"Error loading image encoder: {e}")
            # Handle error appropriately, maybe raise it
            raise e
    return loaded_models["image_encoder"]

def load_text_emoji_pipeline():
    """Loads the basic SDXL pipeline for text-to-emoji."""
    if loaded_models["text_pipe"] is None:
        print("Loading Text-to-Emoji Pipeline (SDXL Base)...")
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                variant="fp16" if TORCH_DTYPE == torch.float16 else None
            ).to(DEVICE)
            # Optional memory savings:
            # pipe.enable_model_cpu_offload()
            loaded_models["text_pipe"] = pipe
            print("Text-to-Emoji Pipeline loaded.")
        except Exception as e:
            print(f"Error loading text pipeline: {e}")
            raise e
    return loaded_models["text_pipe"]

def load_face_style_pipeline():
    """Loads the pipeline with multiple IP-Adapters (Plus and Face)."""
    if loaded_models["face_style_pipe"] is None:
        print("Loading Multi IP-Adapter Pipeline (Face + Style)...")
        encoder = load_image_encoder() # Ensure encoder is loaded first
        if encoder is None: return None # Handle encoder load failure

        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                BASE_MODEL_ID,
                image_encoder=encoder,
                torch_dtype=TORCH_DTYPE,
                variant="fp16" if TORCH_DTYPE == torch.float16 else None,
            ).to(DEVICE)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            adapter_weights = [ADAPTER_WEIGHT_PLUS, ADAPTER_WEIGHT_FACE]
            print(f"Loading IP-Adapter weights: {adapter_weights}...")
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder=IP_ADAPTER_SUBFOLDER,
                weight_name=adapter_weights
            )
            # Optional memory savings:
            # pipe.enable_model_cpu_offload()
            loaded_models["face_style_pipe"] = pipe
            print("Multi IP-Adapter Pipeline loaded.")
        except Exception as e:
            print(f"Error loading face+style pipeline: {e}")
            raise e
    return loaded_models["face_style_pipe"]

def load_text_style_pipeline():
    """Loads the pipeline with a single IP-Adapter (Plus) for style."""
    if loaded_models["text_style_pipe"] is None:
        print("Loading Single IP-Adapter Pipeline (Text + Style)...")
        encoder = load_image_encoder() # Ensure encoder is loaded first
        if encoder is None: return None # Handle encoder load failure

        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                BASE_MODEL_ID,
                image_encoder=encoder,
                torch_dtype=TORCH_DTYPE,
                variant="fp16" if TORCH_DTYPE == torch.float16 else None,
            ).to(DEVICE)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            print(f"Loading IP-Adapter weights: {ADAPTER_WEIGHT_PLUS}...")
            pipe.load_ip_adapter(
                IP_ADAPTER_REPO,
                subfolder=IP_ADAPTER_SUBFOLDER,
                weight_name=ADAPTER_WEIGHT_PLUS # Only load the plus adapter
            )
            # Optional memory savings:
            # pipe.enable_model_cpu_offload()
            loaded_models["text_style_pipe"] = pipe
            print("Single IP-Adapter (Style) Pipeline loaded.")
        except Exception as e:
            print(f"Error loading text+style pipeline: {e}")
            raise e
    return loaded_models["text_style_pipe"]

def get_pipeline_for_mode(mode):
    """Returns the pipeline for the given mode, ensuring only it is on GPU."""
    if mode == "text":
        pipe = load_text_emoji_pipeline()
        ensure_only_one_on_gpu("text_pipe")
        return pipe
    elif mode == "face_style":
        pipe = load_face_style_pipeline()
        ensure_only_one_on_gpu("face_style_pipe")
        return pipe
    elif mode == "text_style":
        pipe = load_text_style_pipeline()
        ensure_only_one_on_gpu("text_style_pipe")
        return pipe
    else:
        return None

def get_pipelines():
    """(Legacy) Loads all pipelines (not recommended for low VRAM)."""
    load_text_emoji_pipeline()
    load_face_style_pipeline()
    load_text_style_pipeline()
    # Return only the pipeline objects, not the encoder separately unless needed
    return {
        "text": loaded_models["text_pipe"],
        "face_style": loaded_models["face_style_pipe"],
        "text_style": loaded_models["text_style_pipe"],
    }

def cleanup_models():
    """Releases model memory (useful for efficient resource management)."""
    global loaded_models
    print("Cleaning up loaded models...")
    del loaded_models["text_pipe"]
    del loaded_models["face_style_pipe"]
    del loaded_models["text_style_pipe"]
    del loaded_models["image_encoder"]
    loaded_models = { k: None for k in loaded_models } # Reset dictionary
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Models cleaned up.")