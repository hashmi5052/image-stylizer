**🖼️ Image Stylizer**

An AI-based app to stylize a base image using a reference style image and powerful LoRA models, supporting both local and Replicate API models!
Built with Gradio, Diffusers, LoRA, and PyTorch.


✨ Features

1- Upload a base image and a style image.

2- Select from local models or Replicate-hosted models.

3- Apply LoRA fine-tuned styles automatically.

4- Output high-quality stylized images.

5- Save results to output/ directory.

6- Run on CPU or GPU locally.

7- Simple drag-and-drop web interface (Gradio).



## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-stylizer.git
cd image-stylizer
```

### 2. (Optional) Create and Activate a Virtual Environment

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

- Copy the example environment file:
  
```bash
cp .env.example .env
```
- Then open `.env` and add your **Replicate API Token** (if using Replicate models).

  
🚀 How to Run

## 🚀 How to Run

### 1. Launch the Gradio App

After installing the dependencies and setting up the environment, simply run:

```bash
python main/main.py
```

The app will start and open in your web browser automatically at:

```
http://127.0.0.1:7860
```

---

### 2. Alternatively

If you are on **Windows**, you can also double-click:

```
run_app.bat
```

If you are on **Mac/Linux**, you can run:

```bash
bash run_app.sh
```

---

### 3. Upload and Stylize

- Upload a **Base Image** and a **Style Image**.
- Select your **Model** (Local or Replicate).
- Adjust the **style parameters** (strength, prompts, etc.).
- Click **Submit** to generate your stylized output image!

The final output will be saved automatically inside the `output/` folder.





📌 Notes

-Local Models: Place your HuggingFace Diffusers format models inside models/.

-LoRA Weights: Place LoRA .safetensors inside lora/.

-Supports running models on CPU (for lower-end machines) and GPU (for faster processing).

-Replicate API is optional — local models will work even without it!


