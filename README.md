## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT:

With the rise of generative AI models, there is a need for user-friendly platforms that allow non-technical users to interact with powerful models like Stable Diffusion. The challenge lies in integrating such models with accessible and intuitive UI frameworks for real-time image generation, prompt-based control, and qualitative evaluation of results. This prototype aims to bridge that gap by building a deployable application using Gradio.

### DESIGN STEPS

#### STEP 1: Model Setup
- Install Stable Diffusion and dependencies (e.g., `diffusers`, `transformers`, `torch`).
- Load the pre-trained model with GPU support.

#### STEP 2: Gradio UI Design
- Create a Gradio interface with:
  - Prompt input
  - Optional negative prompt
  - Sliders for steps, guidance scale
  - Generate button and image output

#### STEP 3: Integration & Deployment
- Connect the UI to the Stable Diffusion generation function.
- Launch the app using `gr.Interface.launch()`.
- (Optional) Deploy online using `share=True` or on platforms like Hugging Face Spaces.

### PROGRAM:
```py
import os
import io
import IPython.display
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
# Helper function
import requests, json

#Text-to-image endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_TTI_BASE']):
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }   
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
    return json.loads(response.content.decode("utf-8"))
import gradio as gr 

#A helper function to convert the PIL image to base64 
# so you can send it to the API
def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def generate(prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    
    output = get_completion(prompt, params)
    pil_image = base64_to_pil(output)
    return pil_image
    gr.Markdown("# Image Generation with Stable Diffusion")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Your prompt") #Give prompt some real estate
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Submit") #Submit button side by side!
    with gr.Accordion("Advanced options", open=False): #Let's hide the advanced options!
            negative_prompt = gr.Textbox(label="Negative prompt")
            with gr.Row():
                with gr.Column():
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                      info="In many steps will the denoiser denoise the image?")
                    guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7,
                      info="Controls how much the text prompt influences the result")
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
                    height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
    output = gr.Image(label="Result") #Move the output up too
            
    btn.click(fn=generate, inputs=[prompt,negative_prompt,steps,guidance,width,height], outputs=[output])

gr.close_all()
demo.launch(share=True, server_port=int(os.environ['PORT4']))
gr.close_all()
```
### OUTPUT:
#### PROMPT:
![Screenshot 2025-05-17 104609](https://github.com/user-attachments/assets/3ebc267d-f243-4e7d-8464-dd3bc476e9f8)

#### ADVANCED OPTIONS:
![Screenshot 2025-05-17 104722](https://github.com/user-attachments/assets/97334c2c-9eb7-4a79-b3f3-9fafaa82c85d)

#### IMAGE:
![image](https://github.com/user-attachments/assets/8e5a7d29-7456-4671-a651-d44d6e3049ee)

### RESULTS  
The prototype successfully generates AI images from text prompts with adjustable parameters using a stable diffusion and responsive Gradio interface.
