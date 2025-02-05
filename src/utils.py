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
import asyncio
import json


sign_dir = "data/signatures/"  # Directory to save audio signatures


# Function to record audio
async def record_audio(filename: str, duration: int):
    """
    Records audio and saves it in 16-bit, 16-kHz, mono format.

    Args:
        filename (str): The name of the file to save the audio to (with .wav extension).
        duration (int): The duration of the recording in seconds.
    """
    # Audio format parameters
    sample_rate = 16000  # 16 kHz
    channels = 1  # Mono

    print(f"Speak into the mic for {duration} seconds:")
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
async def sign(user: str, dur: int = 4):
    """
    Record and save audio as "<user>.wav"

    Args:
        user (str): The name of the user.
        dur (int): The duration of the recording in seconds. Default is 5 seconds.
    """
    save_path = sign_dir + f"{user.lower()}.wav"
    record_audio(save_path, dur)

    print(f"Audio signature saved as {user.lower()}.wav in {sign_dir}")


# Convert audio file to wav format
async def convert_to_wav(
    file_path: str, target_sample_rate: str = "16000", target_channels: str = "1"
):
    """
    Convert an audio file to .wav format using ffmpeg and delete the original file.

    Args:
        file_path (str): The path to the audio file to convert.
    """
    try:
        output_path = str(Path(file_path).with_suffix(".wav"))
        process = asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i",
            file_path,
            "-ar",
            target_sample_rate,
            "-ac",
            target_channels,
            output_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.communicate()
        Path(file_path).unlink()
    except subprocess.CalledProcessError as e:
        print(f"Error converting {file_path} to .wav using ffmpeg: {e}")


# Convert all audio files in a directory to wav format
async def convert_all_to_wav(
    audio_path: str = "known_speakers/audio_files",
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
            await convert_to_wav(str(audio_file))
        elif audio_file.suffix != ".wav":
            # Delete non-wav audio files
            Path(audio_file).unlink()
            print(f"Deleted non-wav file: {audio_file}")


async def combine_audio(
    audio_file_paths: List[str], output_name: str = "combined_audio"
) -> str:
    """
    Combine multiple audio files into a single audio file.

    Args:
        audio_file_paths (List[str]): List of paths to the audio files to combine.
        output_name (str): Name of the output audio file.

    Returns:
        str: Path to the combined audio file.
    """
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


import json
import os


async def speaker_map_processor(
    audio_file: str,
    signatures_path: str = "known_speakers/audio_files/",
    speakers_json: str = "known_speakers/speaker_maps.json",
    output: str = "combined_audio",
) -> (dict, str):
    """
    Combine all signature files and append the final audio buffer into a single .wav file.
    The audio buffer is first converted to .wav before concatenation.

    Args:
        audio_file (str): Path to the input audio file
        signatures_path (str): Path to signatures directory
        speakers_json (str): Path to the speaker maps JSON file
        output (str): Name of the output .wav file

    Returns:
        dict: A dictionary mapping speaker IDs to speaker names
        str: Path to the combined audio file
    """
    # Path to the combined signature file
    combined_signs_path = ".build/combined_signs.wav"

    # Check if the speaker maps already exist and are up to date
    recreate_maps = True
    if os.path.exists(speakers_json):
        try:
            with open(speakers_json, "r") as f:
                speaker_maps = json.load(f)
                if (
                    isinstance(speaker_maps, dict)
                    and len(speaker_maps) == len(os.listdir(signatures_path))
                    and os.path.exists(combined_signs_path)
                ):
                    print("Speaker maps already exist.")
                    recreate_maps = False
                else:
                    print(
                        "Speaker maps are outdated or invalid. Recreating speaker maps."
                    )
        except (json.JSONDecodeError, ValueError):
            print("Speaker maps file is corrupted. Recreating speaker maps.")
    else:
        print("Speaker maps do not exist. Creating speaker maps.")

    if recreate_maps:
        # Convert all signature files to WAV format (if needed)
        await convert_all_to_wav(signatures_path)

        # Get sorted list of signature files
        sign_files = sorted(os.listdir(signatures_path))

        # Combine all signature files into one audio array using combine_audio
        await combine_audio(
            [os.path.join(signatures_path, sign_file) for sign_file in sign_files],
            output_name="combined_signs",
        )

        # Create a speaker map
        speaker_maps = {
            i + 1: os.path.splitext(sign_file)[0].title()
            for i, sign_file in enumerate(sign_files)
        }

        # Save the updated speaker map
        with open(speakers_json, "w") as f:
            json.dump(speaker_maps, f, indent=4)

    # Use the combine_audio function to combine the signature files and the audio buffer
    await combine_audio([combined_signs_path, audio_file], output_name=output)
    final_output_path = f".build/{output}.wav"

    return speaker_maps, final_output_path


async def parse_speaker_text(json_data, speaker_maps):
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
        offset = phrase.get("offsetMilliseconds", 0)
        duration = phrase.get("durationMilliseconds", 0)
        try:
            parsed_output.append(
                f'Speaker: {speaker_maps[str(speaker)]}\nText: "{text}"\nOffset: {offset/1000}\nDuration: {duration/1000}'
            )
        except KeyError:
            parsed_output.append(
                f'Speaker: {speaker}\nText: "{text}"\nOffset: {offset/1000}\nDuration: {duration/1000}'
            )
    return parsed_output


async def get_wav_duration(file_path):
    """
    Returns the duration in seconds of a WAV file.

    :param file_path: Path to the WAV file
    :return: Duration in seconds (float)
    """
    with wave.open(file_path, "r") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration


if __name__ == "__main__":
    # convert_all_to_wav(
    #     "/Users/sam/Desktop/Projects/GitHub Hosted/memoro/server/known_speakers/audio_files"
    # )
    # combine_audio(
    #     ["audio/big_crowd.wav", "audio/heavy_crowd.wav"], "big_crowd"
    # )

    # asyncio.run(record_audio("audio/test.wav", 5))
    print(asyncio.run(get_wav_duration(".build/combined_signs.wav")))
