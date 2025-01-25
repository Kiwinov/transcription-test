import sounddevice as sd
import numpy as np
import wave
import json
from pathlib import Path
from scipy.signal import resample
from scipy.io.wavfile import write
import soundfile as sf
import os
import subprocess
import soundfile as sf

sign_dir = "data/signatures/"  # Directory to save audio signatures
sign_json = Path("data/sign.json")  # JSON file to store user signatures


# Function to record audio
def record_audio(filename: str, duration: int):
    """
    Records audio and saves it in 16-bit, 16-kHz, mono format.

    Args:
        filename (str): The name of the file to save the audio to (with .wav extension).
        duration (int): The duration of the recording in seconds.
    """
    # Audio format parameters
    sample_rate = 16000  # 16 kHz
    channels = 1  # Mono

    print("Speak into the mic for 10 seconds:")
    # Record audio data
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
    )
    sd.wait()  # Wait until the recording is complete
    print("Recording finished.")

    # Save the audio to a .wav file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit is 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


# Function to record and save signature audio file
def sign(user: str, dur: int = 10):
    """
    Record and save audio as "<user>.wav"

    Args:
        user (str): The name of the user.
        dur (int): The duration of the recording in seconds. Default is 10 seconds.
    """
    save_path = sign_dir + f"{user.lower()}.wav"
    record_audio(save_path, dur)

    # Load existing data from sign.json
    sign_data = {}

    if sign_json.exists():
        with sign_json.open("r") as json_file:
            sign_data = json.load(json_file)

    # Append new user and save path
    sign_data[user] = save_path

    # Save updated data back to sign.json
    with sign_json.open("w") as json_file:
        json.dump(sign_data, json_file, indent=4)
    print(f"Audio signature saved as {user.lower()}.wav in {sign_dir}")


# Function to check if all users in json file have signatures
def check_signatures(json_path: Path = sign_json):
    """
    Check if all users in the sign.json file have corresponding signature files.

    Args:
        json_path (Path): The path to the sign.json file. Default is sign_json.
    """
    # Load existing data from sign.json
    sign_data = {}

    if sign_json.exists():
        with sign_json.open("r") as json_file:
            sign_data = json.load(json_file)
    else:
        print("No signature data found.")
        return

    # Check if all users have signatures
    users_to_delete = []
    for user, path in sign_data.items():
        if not Path(path).exists():
            print(f"Signature missing for {user} at {path}")
            users_to_delete.append(user)

    # Remove users with missing signatures
    for user in users_to_delete:
        del sign_data[user]

    # Save updated data back to sign.json
    with sign_json.open("w") as json_file:
        json.dump(sign_data, json_file, indent=4)

    print("Signature check complete.")


def convert_to_wav(
    file_path: str, target_sample_rate: str = "16000", target_channels: str = "1"
):
    """
    Convert an audio file to .wav format using ffmpeg and delete the original file.

    Args:
        file_path (str): The path to the audio file to convert.
    """
    try:
        # Define the output path
        output_path = str(Path(file_path).with_suffix(".wav"))

        # Use ffmpeg to preprocess and convert the file
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                file_path,
                "-ar",
                target_sample_rate,
                "-ac",
                target_channels,
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,  # Suppress stdout
            stderr=subprocess.DEVNULL,  # Suppress stderr
        )
        print(f"Converted {file_path} to {output_path} with 16000Hz, 1 channel.")

        # Delete the original file
        Path(file_path).unlink()
        print(f"Deleted the original file: {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {file_path} to .wav using ffmpeg: {e}")


def convert_all_to_wav(
    audio_path: str = "data/audio/",
):
    """
    Convert all audio files in a directory to .wav format with specified configurations.

    Args:
        audio_path (str): The directory containing the audio files.
        target_sample_rate (int): Target sample rate in Hz. Default is 16000 Hz.
        target_channels (int): Target number of channels. Default is 1 (Mono).
    """
    # Supported audio file extensions
    audio_formats = [".mp3", ".m4a", ".flac", ".ogg"]

    # Iterate over all files in the directory
    for audio_file in Path(audio_path).glob("*"):
        if audio_file.suffix in audio_formats:
            convert_to_wav(str(audio_file))


if __name__ == "__main__":
    convert_all_to_wav()
