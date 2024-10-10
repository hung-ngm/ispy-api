import base64
import io
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import requests
import asyncio
from lmnt.api import Speech
from dotenv import load_dotenv

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

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    # Read and encode the uploaded image
    contents = await file.read()
    base64_image = encode_image(io.BytesIO(contents))

    # Process image with OpenAI
    color_name = process_image_with_openai(base64_image)

    # Generate audio with LMNT
    audio_data = await generate_audio_with_lmnt(color_name)

    # Save audio data to a temporary file
    with open("temp_audio.mp3", "wb") as audio_file:
        audio_file.write(audio_data)

    # Return the audio file
    return FileResponse("temp_audio.mp3", media_type="audio/mpeg", filename="color_name.mp3")
