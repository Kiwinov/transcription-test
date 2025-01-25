import os
import time
import azure.cognitiveservices.speech as speechsdk
import scipy.io.wavfile as wavfile
from pathlib import Path

# Mapping of Azure speaker IDs to actual speaker names
azure_to_name_mapping = {}


def load_speaker_signatures(directory="data/signatures"):
    """Load speaker audio files from the specified directory.

    Args:
        directory (str): The path to the directory containing speaker audio files.

    Returns:
        dict: A mapping of audio file paths to speaker names.
    """
    speaker_mapping = {}
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist.")

    for file_path in directory_path.glob("*.wav"):
        speaker_name = (
            file_path.stem
        )  # Use the file name (without extension) as the speaker name
        speaker_mapping[str(file_path)] = speaker_name

    print(f"Loaded speaker signatures: {speaker_mapping}")
    return speaker_mapping


def play_audio(file_path):
    """Play a given audio file using sounddevice."""
    fs, audio_data = wavfile.read(file_path)  # Read the audio file
    print(f"Playing audio for {Path(file_path).stem}...")


def conversation_transcriber_transcribed_cb(evt, speaker_mapping):
    """Callback for handling transcription and speaker ID mapping."""
    global azure_to_name_mapping
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        speaker_id = evt.result.speaker_id
        if speaker_id not in azure_to_name_mapping:
            # If Azure's speaker ID is new, assign it to the next speaker name
            if speaker_mapping:
                actual_name = list(speaker_mapping.values()).pop(0)
                azure_to_name_mapping[speaker_id] = actual_name
                print(f"Assigned Azure's {speaker_id} to {actual_name}.")
        else:
            actual_name = azure_to_name_mapping[speaker_id]
        print(f"\n{actual_name}: {evt.result.text}\n")
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print("\nNo Match: Speech could not be transcribed.", end="", flush=True)


def recognize_from_microphone(speaker_mapping):
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get("SPEECH_KEY"),
        region=os.environ.get("SPEECH_REGION"),
    )
    speech_config.speech_recognition_language = "en-US"

    # Enable speaker diarization
    speech_config.set_property(
        property_id=speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults,
        value="false",
    )

    # Use the microphone as the audio input
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    # Create a conversation transcriber
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config, audio_config=audio_config
    )

    # Bind the callback for transcription
    conversation_transcriber.transcribed.connect(
        lambda evt: conversation_transcriber_transcribed_cb(evt, speaker_mapping)
    )

    # Play pre-recorded audio to initialize speaker mapping
    print("Playing audio files to identify speakers...")
    for file_path in speaker_mapping:
        play_audio(file_path)

    # Start continuous transcription
    conversation_transcriber.start_transcribing_async()
    print("\nTranscriber Started. Speak into the mic.\n Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping transcription...")
    conversation_transcriber.stop_transcribing_async()


# Main entry point
if __name__ == "__main__":
    try:
        speaker_mapping = (
            load_speaker_signatures()
        )  # Load speaker audio from the signatures directory
        recognize_from_microphone(speaker_mapping)
    except Exception as err:
        print(f"Encountered an exception: {err}")
