import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform


def transcribe_audio():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Real-time speech transcription using Whisper and SpeechRecognition")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], default="medium",
                        help="Specify the model size for Whisper")
    parser.add_argument("--use_non_english", action='store_true',
                        help="Utilize a non-English model for transcription")
    parser.add_argument("--energy_threshold", type=int, default=1000,
                        help="Microphone energy threshold for detecting speech")
    parser.add_argument("--record_interval", type=float, default=2.0,
                        help="Interval in seconds for how often the microphone records")
    parser.add_argument("--pause_duration", type=float, default=3.0,
                        help="Duration in seconds of silence that ends a transcription phrase")
    
    # Special handling for Linux microphone setup
    if platform.startswith('linux'):
        parser.add_argument("--mic_name", type=str, default='pulse',
                            help="Default microphone for Linux. Use 'list' to show available microphones")
    
    args = parser.parse_args()

    # Initialize key variables
    last_phrase_time = None
    audio_queue = Queue()
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = args.energy_threshold
    recognizer.dynamic_energy_threshold = False

    # Setup microphone based on platform
    if platform.startswith('linux'):
        mic_device = None
        if args.mic_name == 'list':
            print("Available microphone devices:")
            for idx, mic in enumerate(sr.Microphone.list_microphone_names()):
                print(f"{idx}: {mic}")
            return
        else:
            for idx, mic in enumerate(sr.Microphone.list_microphone_names()):
                if args.mic_name in mic:
                    mic_device = sr.Microphone(device_index=idx, sample_rate=16000)
                    break
    else:
        mic_device = sr.Microphone(sample_rate=16000)

    # Load Whisper model
    model_name = args.model + (".en" if args.model != "large" and not args.use_non_english else "")
    audio_model = whisper.load_model(model_name)

    # Parameters for audio processing
    record_interval = args.record_interval
    pause_duration = args.pause_duration
    transcription_texts = ['']

    # Adjust microphone settings for ambient noise
    with mic_device:
        recognizer.adjust_for_ambient_noise(mic_device)

    def audio_callback(_, audio_data: sr.AudioData):
        """Callback for processing audio data in the background."""
        raw_audio = audio_data.get_raw_data()
        audio_queue.put(raw_audio)

    # Start recording in the background
    recognizer.listen_in_background(mic_device, audio_callback, phrase_time_limit=record_interval)
    print("Whisper model loaded and ready for transcription.\n")

    while True:
        try:
            current_time = datetime.utcnow()

            if not audio_queue.empty():
                phrase_ended = False

                if last_phrase_time and current_time - last_phrase_time > timedelta(seconds=pause_duration):
                    phrase_ended = True
                
                last_phrase_time = current_time

                # Combine and process the audio data
                combined_audio = b''.join(list(audio_queue.queue))
                audio_queue.queue.clear()

                # Convert audio data to a format Whisper can transcribe
                audio_array = np.frombuffer(combined_audio, dtype=np.int16).astype(np.float32) / 32768.0
                transcription_result = audio_model.transcribe(audio_array, fp16=torch.cuda.is_available())
                transcribed_text = transcription_result['text'].strip()

                if phrase_ended:
                    transcription_texts.append(transcribed_text)
                else:
                    transcription_texts[-1] = transcribed_text

                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription_texts:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\nFinal Transcription:")
    for line in transcription_texts:
        print(line)


if __name__ == "__main__":
    transcribe_audio()
