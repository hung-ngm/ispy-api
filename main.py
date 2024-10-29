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

def preprocess_image(image_bytes):
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

    return processed_image

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

def compare_images_with_openai(base64_image1, base64_image2):
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
                        "text": "I will provide you with two images. Each image contains one main object. Compare the main colour of the main object in both images and tell me if they are the same colour. Output 'The 2 images have same colour.' if they are the same colour and 'The 2 images don't have same colour. First image's colour is {COLOUR OF FIRST IMAGE} and second's image colour is {COLOUR OF SECOND IMAGE}' if they are different."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    return response_json['choices'][0]['message']['content'].strip()

# Function to generate audio with LMNT
# async def generate_audio_with_lmnt(text):
#     async with Speech() as speech:
#         synthesis = await speech.synthesize(text, 'lily')
#     return synthesis['audio']


@app.post("/process-image/")
async def process_image(request: Request):
    # Read the raw bytes from the request body
    image_bytes = await request.body()
    
    # Convert bytes to PIL Image for preprocessing
    processed_image = preprocess_image(image_bytes)
    
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


@app.post("/compare-images/")
async def compare_images(request: Request):
    # Read the JSON payload from the request
    data = await request.json()
    base64_image1 = data['image1']
    base64_image2 = data['image2']

    # Decode base64 images to bytes
    image_bytes1 = base64.b64decode(base64_image1)
    image_bytes2 = base64.b64decode(base64_image2)

    # Preprocess the images
    processed_image1 = preprocess_image(image_bytes1)
    processed_image2 = preprocess_image(image_bytes2)

    # Convert processed images to bytes
    img_byte_arr1 = io.BytesIO()
    processed_image1.save(img_byte_arr1, format='JPEG')
    processed_image_bytes1 = img_byte_arr1.getvalue()
    base64_processed_image1 = base64.b64encode(processed_image_bytes1).decode('utf-8')

    img_byte_arr2 = io.BytesIO()
    processed_image2.save(img_byte_arr2, format='JPEG')
    processed_image_bytes2 = img_byte_arr2.getvalue()
    base64_processed_image2 = base64.b64encode(processed_image_bytes2).decode('utf-8')

    # Compare images with OpenAI API
    result = compare_images_with_openai(base64_processed_image1, base64_processed_image2)

    print(f"Comparison result: {result}")

    return result