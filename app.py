import os
import io
import base64
import random
from flask import Flask, render_template, request, jsonify
from PIL import Image
import diffusers
from utils.model_loader import get_pipeline_for_mode, cleanup_models 
from utils.image_utils import remove_background, load_style_images, load_image_from_path 
from utils.config import STYLE_LIBRARIES, NUM_STYLE_IMAGES_PER_LIBRARY
from generators.text_emoji import generate_text_emoji
from generators.face_style_emoji import generate_face_style_emoji
from generators.text_style_emoji import generate_text_style_emoji


app = Flask(__name__)

# --- Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    # Pass available style names to the template if needed (dynamic select)
    # available_styles = list(STYLE_LIBRARIES.keys())
    return render_template('index.html') # Renders templates/index.html

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint to handle emoji generation requests."""
    # --- Get Data from Request ---
    try:
        mode = request.form.get('mode', 'text')
        prompt = request.form.get('prompt', '')
        negative_prompt = request.form.get('negative_prompt', None) # Use None if empty
        style_name = request.form.get('style', list(STYLE_LIBRARIES.keys())[0] if STYLE_LIBRARIES else None)
        face_image_file = request.files.get('face_image', None)
        # NEW: Get slider values
        guidance_scale = float(request.form.get('guidance_scale', 5.0))
        style_scale = float(request.form.get('style_scale', 0.4))
        face_scale = float(request.form.get('face_scale', 0.7))

        print(f"\nReceived generation request:")
        print(f"  Mode: {mode}")
        print(f"  Prompt: '{prompt}'")
        print(f"  Negative Prompt: '{negative_prompt}'")
        print(f"  Style Name: {style_name}")
        print(f"  Face Image Provided: {'Yes' if face_image_file else 'No'}")
        print(f"  Guidance Scale: {guidance_scale}")
        print(f"  Style Scale: {style_scale}")
        print(f"  Face Scale: {face_scale}")

        # --- Input Validation ---
        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        generated_image = None
        seed = random.randint(0, 2**32 - 1) # Generate random seed for variation
        print(f"  Using Seed: {seed}")

        # --- Mode-Specific Logic ---
        pipeline_to_use = get_pipeline_for_mode(mode)
        if not pipeline_to_use:
            return jsonify({"error": f"{mode.capitalize()} generation model not loaded."}), 500

        if mode == 'text':
            # Add emoji style suffix to prompt
            emoji_style_suffix = (
                "Apple Emoji Style, Plain Matte White background, centered, minimal, no text, "
                "no watermark, no border, high contrast, simple, clean, isolated, icon, 3D, soft shadow"
            )
            full_prompt = f"{prompt.strip()}, {emoji_style_suffix}"
            generated_image = generate_text_emoji(
                pipe=pipeline_to_use,
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                seed=seed
            )

        elif mode == 'text_style':
            if not style_name or style_name not in STYLE_LIBRARIES:
                return jsonify({"error": f"Invalid style selected: {style_name}"}), 400

            style_dir = STYLE_LIBRARIES[style_name]
            style_images = load_style_images(style_dir, NUM_STYLE_IMAGES_PER_LIBRARY)
            if not style_images:
                return jsonify({"error": f"Could not load style images from '{style_dir}'."}), 500

            generated_image = generate_text_style_emoji(
                pipe=pipeline_to_use,
                style_images=style_images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                style_scale=style_scale,
                guidance_scale=guidance_scale,
                seed=seed
            )

        elif mode == 'face_style':
            if not face_image_file:
                return jsonify({"error": "Face image is required for Face+Style mode."}), 400
            if not style_name or style_name not in STYLE_LIBRARIES:
                return jsonify({"error": f"Invalid style selected: {style_name}"}), 400

            # Load face image from upload
            try:
                face_image = Image.open(face_image_file.stream).convert("RGB")
            except Exception as e:
                print(f"Error reading uploaded face image: {e}")
                return jsonify({"error": "Invalid or corrupted face image file."}), 400

            # Load style images
            style_dir = STYLE_LIBRARIES[style_name]
            style_images = load_style_images(style_dir, NUM_STYLE_IMAGES_PER_LIBRARY)
            if not style_images:
                return jsonify({"error": f"Could not load style images from '{style_dir}'."}), 500

            generated_image = generate_face_style_emoji(
                pipe=pipeline_to_use,
                face_image=face_image,
                style_images=style_images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                style_scale=style_scale,
                face_scale=face_scale,
                guidance_scale=guidance_scale,
                seed=seed
            )

        else:
            return jsonify({"error": f"Invalid mode specified: {mode}"}), 400

        # --- Process Result ---
        if generated_image is None:
            print("Generation function returned None.")
            return jsonify({"error": "Failed to generate image. Check server logs for details."}), 500

        print("Generation successful. Removing background...")
        # Remove background
        final_image = remove_background(generated_image)
        if final_image is None:
            print("Warning: Background removal failed. Returning original image.")
            final_image = generated_image # Fallback to original if removal fails

        # Convert final image to Base64 PNG data URL
        buffered = io.BytesIO()
        final_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_url = f"data:image/png;base64,{img_str}"

        print("Background removal complete. Sending image data.")
        return jsonify({"image_data": image_data_url})

    except Exception as e:
        print(f"Unhandled error during generation: {e}")
        import traceback
        traceback.print_exc() # Log the full traceback for debugging
        return jsonify({"error": "An unexpected server error occurred."}), 500



if __name__ == '__main__':
    
  
    app.run(debug=True, host='0.0.0.0', port=5000)

