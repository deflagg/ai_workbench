import openai
from langchain_core.tools import tool

@tool
def tts_whisper(text: str) -> str:
    """
    Converts text to speech using OpenAI's Text-to-Speech API.
     
    Args:
        text (str): The text to convert to speech.
        
    Returns:
        str: The speech generated from the input text.
        
    Examples:
        >>> tts_whisper("Hello, how are you?")
        "Hello, how are you?"
    """

    response = openai.audio.speech.create(
        model = "tts-1",
        voice = "alloy",
        input = text
    )
    
    response.write_to_file("./output.mp3")
    
    return "./output.mp3"
