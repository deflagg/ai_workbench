import groq
import pyaudio
import numpy as np

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the microphone stream
audio_stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

# Create a GROQ client
#client = groq.Client("whisper", "en", "cpu")

# Create a GROQ stream
#groq_stream = client.stream()

# Silence threshold
SILENCE_THRESHOLD = 500  # Adjust this value as needed

# Stream audio to Whisper API using GROQ
try:
    while True:
        # Read audio data from the microphone stream
        audio_data = audio_stream.read(1024)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Check for silence
        if np.abs(audio_np).mean() > SILENCE_THRESHOLD:
            # Stream audio data to GROQ
            # groq_stream.write(audio_data)

            # Get the transcription result from GROQ
            #result = groq_stream.result()

            # Print the transcription result
            #print(result.text)
            print(audio_data)
        # else:
        #     print("Silence detected, not streaming to GROQ.")
except KeyboardInterrupt:
    print("Transcription stopped.")

# Close the microphone stream
audio_stream.stop_stream()
audio_stream.close()
p.terminate()
