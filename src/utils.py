import sounddevice as sd
import numpy as np
import wave
from pathlib import Path
from scipy.signal import resample
from scipy.io.wavfile import write
import os
import subprocess
import soundfile as sf
from typing import List


sign_dir = "data/signatures/"  # Directory to save audio signatures


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
def sign(user: str, dur: int = 5):
    """
    Record and save audio as "<user>.wav"

    Args:
        user (str): The name of the user.
        dur (int): The duration of the recording in seconds. Default is 5 seconds.
    """
    save_path = sign_dir + f"{user.lower()}.wav"
    record_audio(save_path, dur)

    print(f"Audio signature saved as {user.lower()}.wav in {sign_dir}")


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
        elif audio_file.suffix != ".wav":
            # Delete non-wav audio files
            Path(audio_file).unlink()
            print(f"Deleted non-wav file: {audio_file}")


def combine_audio(
    audio_file_paths: List[str], output_name: str = "combined_audio"
) -> str:
    if not audio_file_paths:
        raise ValueError("No audio files provided.")

    target_samplerate = 16000  # 16 kHz
    combined_data = None

    for file_path in audio_file_paths:
        # Read the audio file
        data, sr = sf.read(file_path)

        # Normalize audio if it's integer-based
        if np.issubdtype(data.dtype, np.integer):
            data = data / np.iinfo(data.dtype).max

        # Resample if the sample rate is different from the target
        if sr != target_samplerate:
            num_samples = int(len(data) * target_samplerate / sr)
            data = resample(data, num_samples)

        # Concatenate the audio data
        if combined_data is None:
            combined_data = data
        else:
            combined_data = np.concatenate((combined_data, data))

    # Normalize the combined data to prevent clipping
    combined_data = np.clip(combined_data, -1.0, 1.0)

    # Convert to 16-bit PCM
    combined_data = (combined_data * 32767).astype(np.int16)

    # Ensure the output directory exists
    output_dir = ".build"
    os.makedirs(output_dir, exist_ok=True)

    # Save the combined audio as 16-bit 16 kHz WAV
    combined_file_path = os.path.join(output_dir, f"{output_name}.wav")
    write(combined_file_path, target_samplerate, combined_data)
    print(f"Combined audio saved as {combined_file_path}")

    return combined_file_path


def pre_processor(
    audio_file: str,
    audio_path: str = "data/audio/",
    signatures_path: str = "data/signatures/",
    output: str = "combined_final",
):
    """
    Combine all signature files and append the final audio file into a single .wav file

    Args:
        audio_path (str): Path to the directory containing the audio files
        signatures_path (str): Path to signatures directory
        output (str): Name of the output .wav file
        audio_file (str): Name of the audio file to be appended
    """

    # Convert all audio files to .wav format
    utils.convert_all_to_wav(audio_path)
    utils.convert_all_to_wav(signatures_path)

    # Combine all signature files into a single .wav file
    sign_files = sorted([str(sign) for sign in os.listdir(signatures_path)])
    print(sign_files)
    utils.combine_audio(["data/signatures/" + sign_file for sign_file in sign_files])

    speaker_maps = dict()

    for i, sign_path in enumerate(sign_files):
        speaker = sign_path.split(".")[0]
        speaker_maps[speaker] = int(i)

    # Add the audio file that needs to be transcribed to combined audio
    combined_signs = ".build/combined_audio.wav"
    utils.combine_audio([combined_signs, audio_file], output)

    return speaker_maps


def parse_speaker_text(json_data):
    """
    Parse the JSON response to extract speaker and text information.

    Args:
        json_data (dict): The transcription JSON response.

    Returns:
        list: A list of strings with speaker labels and their spoken text.
    """
    parsed_output = []

    # Extract phrases from the response
    phrases = json_data.get("phrases", [])

    for phrase in phrases:
        speaker = phrase.get("speaker", "Unknown")
        text = phrase.get("text", "")
        parsed_output.append(f'Speaker: {speaker}\nText: "{text}"')

    return parsed_output


if __name__ == "__main__":
    convert_all_to_wav("data/signatures")
