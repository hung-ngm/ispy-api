import base64
import io
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to preprocess image
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
    print(f"response: {response}")
    return response.json()['choices'][0]['message']['content'].strip()


def compare_color_with_image(color_name, base64_image):
    try:
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
                            "text": "I will provide you with a color name and an image. The image contains one main object. " + \
                                    "The color name is " + color_name + ". Your task is to analyze the main color of the object in the image and compare it with the provided color name. " + \
                                    "There are three possible cases: " + \
                                    "1. **Exactly Same Color**: If the color of the object in the image is exactly the same as the provided color name, output STRICTLY LIKE 'SAME, colour X, colour Y' in which colour X is {color_name} and colour Y is colour of the image. " + \
                                    "2. **Nearly Same Color**: If the colors are close (e.g., cyan and turquoise), output STRICTLY LIKE 'CLOSE, colour X, colour Y', where colour X is {color_name} and colour Y is the colour of the object in the image. " + \
                                    "3. **Different Color**: If the colors are not close, output STRICTLY LIKE 'DIFFERENT, colour X, colour Y', where colour X is {color_name} and colour Y is the colour of the object in the image. " + \
                                    "Please analyze the image and provide the appropriate response according to the cases above."
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
            "max_tokens": 500
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(f"response: {response}")
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)} 

@app.post("/process-image/")
async def process_image(request: Request):
    # Read the raw bytes from the request body
    image_bytes = await request.body()
    
    # Preprocess the image
    processed_image = preprocess_image(image_bytes)
    
    # Convert processed image back to bytes for OpenAI API
    img_byte_arr = io.BytesIO()
    processed_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Encode the processed image to base64
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
    
    # Process image with OpenAI to get color name
    # color_name = process_image_with_openai(base64_image)
    color_name = process_image_with_openai(base64_image)

    print(f"Color name is: {color_name}")
    
    return {"color_name": color_name}

@app.post("/compare-image/{color}")
async def compare_image(color: str, request: Request):
    try:
        # Read the raw bytes from the request body
        image_bytes = await request.body()

          # Debug: Check if image_bytes is empty or None
        if not image_bytes:
            print("Debug: Received empty image_bytes from request.")
            return {"error": "No image data received in request body."}
        else:
            print(f"Debug: Received image_bytes of length {len(image_bytes)}")
        
        # Preprocess the image (assuming you have this function)
        processed_image = preprocess_image(image_bytes)

        # Convert processed image to bytes
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='JPEG')
        processed_image_bytes = img_byte_arr.getvalue()

           # Debug: Check if processed_image_bytes is empty or None
        if not processed_image_bytes:
            print("Debug: Processed image bytes are empty.")
            return {"error": "Failed to process image."}
        else:
            print(f"Debug: Processed image_bytes of length {len(processed_image_bytes)}")

        # Encode the processed image to base64 for OpenAI API
        base64_processed_image = base64.b64encode(processed_image_bytes).decode('utf-8')

            # Debug: Check if base64_processed_image is empty or None
        if not base64_processed_image:
            print("Debug: Base64 encoded processed image is empty.")
            return {"error": "Failed to encode image to base64."}
        else:
            print(f"Debug: base64_processed_image length {len(base64_processed_image)}")

        # Compare the provided color with the image using OpenAI API
        result = compare_color_with_image(color, base64_processed_image)

        print(f"Comparison result: {result}")

        return {"result": result}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}