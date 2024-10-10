# Image Color to Speech API

This FastAPI application takes an image as input, identifies the main color of an object in the image using OpenAI's GPT-4 Vision model, and then converts the color name to speech using LMNT's text-to-speech API.

## Setup

1. Clone this repository.

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API and LMNT API key as an environment variable:
   ```
   export OPENAI_API_KEY=your_api_key_here
   export LMNT_API_KEY=your_api_key_here
   ```

4. Run the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

## Usage

Send a POST request to `http://localhost:8000/process-image/` with an image file in the request body. The API will return an MP3 file containing the audio of the main color name of the object in the uploaded image.

Example using curl: