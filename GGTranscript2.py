import io
import os
import pyaudio
from google.cloud import speech
from google.api_core.exceptions import GoogleAPIError

# Assurez-vous que GOOGLE_APPLICATION_CREDENTIALS est défini dans votre environnement
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/loren/Desktop/MI/PythonTranscriptionFSM/Client-Key.json"

def stream_generator(rate, chunk):
    audio_interface = pyaudio.PyAudio()
    audio_stream = audio_interface.open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=rate,
                                        input=True,
                                        frames_per_buffer=chunk)

    while True:
        yield audio_stream.read(chunk)

def transcribe_streaming():
    client = speech.SpeechClient()

    # Configure le flux audio
    rate = 16000
    chunk = int(rate / 10)  # 100ms

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=rate,
        language_code="en-US",
    )
    
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    # Utilisation du flux audio
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                for chunk in stream_generator(rate, chunk))

    try:
        # Reconnaissance continue
        responses = client.streaming_recognize(streaming_config, requests)

        for response in responses:
            for result in response.results:
                if result.is_final:
                    print(f"Transcribed Text: {result.alternatives[0].transcript}")

    except GoogleAPIError as e:
        print(f"Error during transcription: {e}")

if __name__ == "__main__":
    transcribe_streaming()
