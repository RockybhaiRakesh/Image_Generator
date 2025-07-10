# 🎨 Stable Diffusion Image Generator using Hugging Face `diffusers`

This Python script uses the `StableDiffusionPipeline` from the Hugging Face 🤗 Diffusers library to generate AI art from a text prompt.

## 📌 Features

- Uses pre-trained model: `runwayml/stable-diffusion-v1-5`
- CPU-compatible (works without GPU)
- Saves output image as `output_image.png`

---

## 🛠️ Prerequisites

Ensure you have Python 3.8 or higher installed.

You can check your Python version with:
```bash
python --version
📦 Installation Steps
1. ⚙️ Create a Virtual Environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate      # For Linux/macOS
venv\Scripts\activate         # For Windows
2. 📥 Install Required Packages
bash
Copy
Edit
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers scipy safetensors
🔐 (Optional) Hugging Face Authentication
If this is your first time using Hugging Face models, you may need to log in:

bash
Copy
Edit
huggingface-cli login
Then paste your token from https://huggingface.co/settings/tokens

⚠️ This step may be required if the model is gated or restricted.

🚀 Run the Script
Create a Python file, e.g., generate.py, and paste the following:

python
Copy
Edit
from diffusers import StableDiffusionPipeline
import torch

# Load pre-trained model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# Use CPU only (set to "cuda" if you have GPU)
pipe.to("cpu")

# Your image prompt
prompt = "creative and colorful abstract painting in wall art style"

# Generate image
image = pipe(prompt).images[0]

# Save image
image.save("output_image.png")
print("✅ Image saved as output_image.png")
Then run:

bash
Copy
Edit
python generate.py
📸 Output
After running, your generated image will be saved as:

Copy
Edit
output_image.png
You can open it with any image viewer.

⚡ GPU Support (Optional)
If you have a CUDA-compatible GPU:

Install the GPU version of PyTorch:

bash
Copy
Edit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Change this line in the script:

python
Copy
Edit
pipe.to("cuda")
🧠 Model Reference
runwayml/stable-diffusion-v1-5

🧼 Troubleshooting
Slow generation? Use GPU if available.

Out of memory on CPU? Try a simpler prompt or reduce resolution.

Permission errors? Run terminal as admin or use venv.

📜 License
This project uses models from Hugging Face under their respective licenses.

👨‍💻 Author
Built with ❤️ by [Your Name]

yaml
Copy
Edit

---

Let me know if you want the README in **PDF format** or with **GPU/CLI prompt arguments** added!