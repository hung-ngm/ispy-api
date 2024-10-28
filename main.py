import base64
import io
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
import requests
import asyncio
from lmnt.api import Speech
from dotenv import load_dotenv
import numpy as np
from PIL import Image

load_dotenv()

app = FastAPI()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to encode the image
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Function to process image with OpenAI
def process_image_with_openai(base64_image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I will provide you some image. The image will contain one main object. You should output STRICTLY JUST the name of the main colour of the object. Be specific about the colour you output."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content'].strip()

# Function to generate audio with LMNT
async def generate_audio_with_lmnt(text):
    async with Speech() as speech:
        synthesis = await speech.synthesize(text, 'lily')
    return synthesis['audio']

# @app.post("/process-image/")
# async def process_image(request: Request):
#     # Read the raw bytes from the request body
#     image_bytes = await request.body()

#     # Encode the image to base64
#     base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
#     # Process image with OpenAI
#     color_name = process_image_with_openai(base64_image)

#     print(f"Color name is: {color_name}")

 
#     return color_name

@app.post("/process-image/")
async def process_image(request: Request):
    # Read the raw bytes from the request body
    image_bytes = await request.body()
    
    # Convert bytes to PIL Image for preprocessing
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert image to numpy array for processing
    image_array = np.array(image)
    
    # Calculate the mean values for each channel
    mean_r = np.mean(image_array[:, :, 0])
    mean_g = np.mean(image_array[:, :, 1])
    mean_b = np.mean(image_array[:, :, 2])
    
    # Calculate the overall mean of all channels
    target_mean = np.mean([mean_r, mean_g, mean_b])
    
    # Calculate adjustment factors
    r_factor = target_mean / mean_r if mean_r > 0 else 1
    g_factor = target_mean / mean_g if mean_g > 0 else 1
    b_factor = target_mean / mean_b if mean_b > 0 else 1
    
    # Apply adjustments with clipping to prevent overflow
    image_array[:, :, 0] = np.clip(image_array[:, :, 0] * r_factor, 0, 255)
    image_array[:, :, 1] = np.clip(image_array[:, :, 1] * g_factor, 0, 255)
    image_array[:, :, 2] = np.clip(image_array[:, :, 2] * b_factor, 0, 255)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(image_array.astype('uint8'))
    
    # Convert processed image back to bytes for OpenAI API
    img_byte_arr = io.BytesIO()
    processed_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Encode the processed image to base64
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
    
    # Process image with OpenAI
    color_name = process_image_with_openai(base64_image)

    print(f"Color name is: {color_name}")
    
    return color_name


