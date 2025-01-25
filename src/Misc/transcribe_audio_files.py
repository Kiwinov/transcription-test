import os
import wave
import azure.cognitiveservices.speech as speechsdk
from pathlib import Path
import json
from sign_manage import check_signatures


def read_signatures(sign_json: Path = Path("data/sign.json")) -> (dict, dict):
    """Read signature file mappings from a JSON file.

    Args:
        sign_json (Path): The path to the JSON file.
    Returns:
        tuple: A tuple containing two dictionaries -
               paths (mapping of speaker names to file paths) and
               maps (mapping of Azure-assigned speaker IDs to names).
    """
    check_signatures(sign_json)
    if sign_json.exists():
        with sign_json.open("r") as json_file:
            raw_data = json.load(json_file)
            paths = raw_data
            maps = {
                f"Guest-{i + 1}": name for i, (name, _) in enumerate(raw_data.items())
            }
        print(f"Debug: Signatures loaded - Paths: {paths}, Maps: {maps}")
        return paths, maps
    return {}, {}


def combine_audio_files(paths, maps, audio_file) -> Path:
    """Combine all audio files into a single file for diarization.

    Args:
        paths: A dictionary mapping speaker names to their audio file paths.
        maps: A dictionary mapping Azure-assigned speaker IDs to names.

    Returns:
        Path: The path to the combined audio file.
    """
    combined_file_path = Path("data/combined_audio.wav")
    with wave.open(str(combined_file_path), "wb") as combined_audio:
        params_set = False
        for i in range(len(maps)):
            guest_key = f"Guest-{i + 1}"
            audio_path = Path(paths[maps[guest_key]])
            if audio_path.exists():
                with wave.open(str(audio_path), "rb") as audio_segment:
                    if not params_set:
                        combined_audio.setparams(audio_segment.getparams())
                        params_set = True
                    combined_audio.writeframes(
                        audio_segment.readframes(audio_segment.getnframes())
                    )
            else:
                print(f"Warning: Audio file {audio_path} does not exist. Skipping.")
        print(f"Debug: Combined audio created at {combined_file_path}")
    return combined_file_path


def process_audio_file(speech_config, combined_audio_path, maps):
    """Send the combined audio file for diarization and process results.

    Args:
        speech_config: The Azure speech configuration object.
        combined_audio_path (Path): The path to the combined audio file.
        maps: A dictionary mapping Azure-assigned speaker IDs to names.

    Returns:
        dict: A mapping of speaker IDs to their names after diarization.
    """
    if not combined_audio_path.exists():
        print(f"Error: Combined audio file {combined_audio_path} does not exist.")
        return {}

    try:
        # Create audio configuration for the combined file
        audio_config = speechsdk.audio.AudioConfig(filename=str(combined_audio_path))

        # Create a transcriber for diarization
        conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config, audio_config=audio_config
        )

        speaker_map = {}

        def diarization_handler(evt):
            """Handle transcription events."""
            print(f"Debug: Diarization event received - {evt}")
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                speaker_id = evt.result.speaker_id
                if speaker_id not in speaker_map:
                    speaker_map[speaker_id] = f"Guest-{len(speaker_map) + 1}"
                transcription_text = evt.result.text.strip()
                if transcription_text:
                    speaker_name = speaker_map[speaker_id]
                    print(f"{speaker_name}: {transcription_text}")

        conversation_transcriber.transcribed.connect(
            lambda evt: diarization_handler(evt)
        )
        print("Info: Starting diarization for the combined audio file...")

        # Start diarization and wait for completion
        start_future = conversation_transcriber.start_transcribing_async()
        start_future.get()  # Wait for transcription to start

        print("Info: Diarization in progress...")

        # Wait for transcription to complete
        stop_future = conversation_transcriber.stop_transcribing_async()
        stop_future.get()  # Wait for transcription to stop

        print("Info: Diarization complete.")

        print(f"Debug: Speaker Map: {speaker_map}")
        return speaker_map

    except Exception as e:
        print(f"Error during diarization: {e}")
        return {}


def main():
    """Main entry point for the script."""
    try:
        # Read the speaker signature mappings
        sign_file = Path("data/sign.json")
        paths, maps = read_signatures(sign_file)

        if not maps:
            print("Error: No speaker signature mapping found in sign.json.")
            return

        # Initialize Azure Speech Configuration
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get("SPEECH_KEY"),
            region=os.environ.get("SPEECH_REGION"),
        )
        speech_config.speech_recognition_language = "en-US"

        # Combine all audio files into a single file
        combined_audio_path = combine_audio_files(
            paths, maps, audio_file="data/audio.wav"
        )

        # Perform diarization on the combined audio file
        speaker_map = process_audio_file(speech_config, combined_audio_path, maps)

        if speaker_map:
            print("Diarization and transcription results:")
            for speaker_id, speaker_name in speaker_map.items():
                print(f"{speaker_id}: {speaker_name}")
        else:
            print("No results found during diarization.")

    except Exception as err:
        print(f"Critical Error: Encountered an exception: {err}")


if __name__ == "__main__":
    main()
