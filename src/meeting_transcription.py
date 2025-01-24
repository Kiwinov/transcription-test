import os
import time
import sounddevice as sd
import numpy as np
import wavio
import azure.cognitiveservices.speech as speechsdk
import scipy.io.wavfile as wavfile

# Mapping of Azure speaker IDs to actual speaker names
azure_to_name_mapping = {}


def record_audio_for_speakers():
    """Prompt user to set up pre-recorded audio for each speaker and record new audio."""
    speaker_mapping = {}
    num_speakers = int(input("Enter the number of speakers: "))
    os.makedirs("./audio", exist_ok=True)

    for i in range(num_speakers):
        name = input(f"Enter name for Speaker {i+1}: ")
        file_path = f"./audio/{name}.wav"
        print(
            f"Recording for {name}. This will save the audio to {file_path}. Press RETURN to stop."
        )
        record_audio(file_path)  # Record the audio and save it to the specified path
        speaker_mapping[file_path] = name
    return speaker_mapping


def record_audio(file_path):
    """Record audio from the microphone and save it as a WAV file."""
    fs = 16000  # Sample rate
    duration = 5  # Duration in seconds

    # Record audio using sounddevice
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished

    # Save the audio as a WAV file using wavio
    wavio.write(file_path, audio_data, fs)

    print("Recording complete.")


def play_audio(file_path):
    """Play a given audio file using sounddevice."""
    # Read the audio file using scipy.io.wavfile
    fs, audio_data = wavfile.read(
        file_path
    )  # fs is the sample rate, audio_data is the actual data


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
        print(f"Playing audio for {speaker_mapping[file_path]}...")
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
        speaker_mapping = record_audio_for_speakers()
        recognize_from_microphone(speaker_mapping)
    except Exception as err:
        print(f"Encountered an exception: {err}")
