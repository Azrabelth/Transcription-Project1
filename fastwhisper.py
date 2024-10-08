import os
import time
import wave
import pyaudio
from faster_whisper import WhisperModel

# Définir les constantes
NEON_GREEN = '\033[32m'
RESET_COLOR = '\033[0m'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Fonction pour enregistrer un fragment audio
def record_chunk(p, stream, file_path, chunk_length=1):
    """
    Enregistre un fragment audio dans un fichier.

    Args:
        p (pyaudio.PyAudio): Objet PyAudio.
        stream (pyaudio.Stream): Flux PyAudio.
        file_path (str): Chemin du fichier où le fragment audio sera enregistré.
        chunk_length (int): Durée du fragment audio en secondes.

    Returns:
        None
    """

    frames = []

    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ''.join(segment.text for segment in segments)
    return transcription

def main2():
    """
    Fonction principale du programme.
    """

    # Sélectionner le modèle Whisper
    model = WhisperModel("medium", device="cuda", compute_type="float16")

    # Initialiser PyAudio
    p = pyaudio.PyAudio()

    # Ouvrir le flux d'enregistrement
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    # Initialiser une chaîne vide pour accumuler les transcriptions
    accumulated_transcription = ""

    try:
        while True:
            # Enregistrer un fragment audio
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)

            # Transcrire le fragment audio
            transcription = transcribe_chunk(model, chunk_file)
            print(NEON_GREEN + transcription + RESET_COLOR)

            # Supprimer le fichier temporaire
            os.remove(chunk_file)

            # Ajouter la nouvelle transcription à la transcription accumulée
            accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        print("Arrêt...")

        # Enregistrer la transcription accumulée dans un fichier journal
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)

    finally:
        print("LOG" + accumulated_transcription)
        # Fermer le flux d'enregistrement
        stream.stop_stream()
        stream.close()

        # Arrêter PyAudio
        p.terminate()


if __name__ == "__main__":
    main2()
