import os
import uuid
import shutil
import gradio as gr
from dotenv import load_dotenv
import replicate
from PIL import Image
from io import BytesIO
import requests
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from safetensors.torch import load_file as load_safetensor

# Load environment variables
load_dotenv()
replicate_token = os.getenv("REPLICATE_API_TOKEN")
replicate.Client(api_token=replicate_token)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# Utility functions
# ------------------------

def get_model_list():
    model_folder = "models"
    return [f.name for f in os.scandir(model_folder) if f.is_dir()]

def get_lora_list():
    lora_folder = "lora"
    return [f.name for f in os.scandir(lora_folder) if f.is_file() and f.name.endswith(".safetensors")]

# ------------------------
# Local pipeline runner
# ------------------------

def run_local_model(input_image, local_model_name, lora_name, style_strength, prompt, negative_prompt, num_inference_steps, guidance_scale, seed):
    model_path = os.path.join("models", local_model_name)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    ).to("cuda")

    if lora_name:
        lora_path = os.path.join("lora", lora_name)
        lora_state_dict = load_safetensor(lora_path)
        pipe.load_lora_weights(lora_state_dict)

    generator = torch.manual_seed(seed)
    input_image = input_image.convert("RGB").resize((512, 512))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        strength=style_strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]

    return result

# ------------------------
# Main inference function
# ------------------------

def stylize(backend, input_image, style_image, local_model_name, lora_name, style_strength, structure_strength, prompt, negative_prompt, num_inference_steps, guidance_scale, seed):
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_input.png")
    style_path = os.path.join(UPLOAD_DIR, f"{file_id}_style.png")

    input_image.save(input_path)
    style_image.save(style_path)

    if backend == "replicate":
        with open(input_path, "rb") as input_file, open(style_path, "rb") as style_file:
            output = replicate.run(
                "philz1337x/style-transfer:a15407d73d9669676d623e37ee3b6d43642439beec1b99639967d215bcf42fc4",
                input={
                    "image": input_file,
                    "image_style": style_file,
                    "style_strength": style_strength,
                    "structure_strength": structure_strength,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed
                }
            )
        response = requests.get(output[0])
        if response.status_code == 200:
            output_image = Image.open(BytesIO(response.content))
            output_path = os.path.join(OUTPUT_DIR, f"{file_id}_output.png")
            output_image.save(output_path)
            return output_image
        else:
            raise Exception("Failed to download image from replicate.")
    else:
        return run_local_model(
            input_image=input_image,
            local_model_name=local_model_name,
            lora_name=lora_name,
            style_strength=style_strength,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )

# ------------------------
# Gradio UI
# ------------------------

gr_interface = gr.Interface(
    fn=stylize,
    inputs=[
        gr.Dropdown(choices=["replicate", "local"], value="replicate", label="Run On"),
        gr.Image(label="Input Image", type="pil"),
        gr.Image(label="Style Image", type="pil"),
        gr.Dropdown(choices=get_model_list(), label="Local Base Model"),
        gr.Dropdown(choices=get_lora_list(), label="LoRA Model", allow_none=True),
        gr.Slider(minimum=0, maximum=3, value=0.4, label="Style Strength", step=0.01),
        gr.Slider(minimum=0, maximum=3, value=0.6, label="Structure Strength", step=0.01),
        gr.Textbox(value="masterpiece, best quality, highres", label="Prompt"),
        gr.Textbox(value="worst quality, low quality, normal quality", label="Negative Prompt"),
        gr.Slider(minimum=1, maximum=100, value=30, label="Number of Inference Steps"),
        gr.Slider(minimum=1, maximum=50, value=8, label="Guidance Scale"),
        gr.Number(value=1337, label="Seed", interactive=True),
    ],
    outputs="image",
    live=False,
    title="Style Transfer App",
    description="Choose backend (local or Replicate), upload an input and style image, and select your model + LoRA to stylize the image."
)

if __name__ == "__main__":
    gr_interface.launch()
