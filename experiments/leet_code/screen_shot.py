import keyboard
import pyautogui
from PIL import Image
import io
import openai
import base64
import sys
import os

sys.path.append('d:/source/vscode/ai_workbench')
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

# Initialize the OpenAI client
client = openai.OpenAI()

def send_to_chatgpt(image_data):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this screenshot. Create a Python script, one method, to solve the problem."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

def on_screenshot():
    # Take a screenshot
    screenshot = pyautogui.screenshot()

    # Convert the screenshot to base64 for sending
    img_byte_arr = io.BytesIO()
    screenshot.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    encoded_string = base64.b64encode(img_byte_arr).decode("utf-8")

    # Send the screenshot to ChatGPT
    chatgpt_response = send_to_chatgpt(encoded_string)
    print(chatgpt_response)  # Or do something else with the response

# Register the hotkey
keyboard.add_hotkey("num lock", on_screenshot)

# Keep the script running
keyboard.wait("esc")  # Press 'esc' to stop the script