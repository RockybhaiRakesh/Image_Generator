from diffusers import StableDiffusionPipeline
import torch

# Load pre-trained model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Use CPU only (set to cuda if you have GPU)
pipe.to("cpu")

# Your image prompt
prompt = (
    "creative and colorful abstract painting in wall art style"
)
# Generate image
image = pipe(prompt).images[0]

# Save image
image.save("output_image.png")
print("✅ Image saved as output_image.png")
