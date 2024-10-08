import io
import os
import pyaudio
from google.cloud import speech
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Chemin vers votre fichier Client_secret.json
CLIENT_SECRET_FILE = "C:/Users/loren/Desktop/MI/PythonTranscriptionFSM/Client_secret.json"

def get_credentials():
    # Charger les informations d'identification OAuth2
    credentials = Credentials.from_authorized_user_file(CLIENT_SECRET_FILE)
    
    # Vérifiez si les jetons sont expirés et rafraîchissez-les si nécessaire
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    
    return credentials

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
    credentials = get_credentials()
    client = speech.SpeechClient(credentials=credentials)

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

    # Reconnaissance continue
    responses = client.streaming_recognize(streaming_config, requests)

    try:
        for response in responses:
            for result in response.results:
                if result.is_final:
                    print(f"Transcribed Text: {result.alternatives[0].transcript}")

    except Exception as e:
        print(f"Error during transcription: {e}")

if __name__ == "__main__":
    transcribe_streaming()
