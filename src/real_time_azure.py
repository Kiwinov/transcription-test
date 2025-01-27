import os
import time
import azure.cognitiveservices.speech as speechsdk


# Transcriber helper functions
def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    # After transcribing, give the result
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        if evt.result.speaker_id:
            print(f"\nSpeaker: {evt.result.speaker_id}: {evt.result.text}")
            print()
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print("\nNo Match: Speech could not be transcribed.", end="", flush=True)


def conversation_transcriber_transcribing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    """This function overwrites the transcription line during ongoing recognition"""
    # Move the cursor to the beginning of the line using \r to overwrite
    print(f"\r{evt.result.speaker_id}: {evt.result.text}", end="", flush=True)


# Utility functions for when we start, cancel or stop the service
def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print("Session Started: {}".format(evt))


def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print("Recognition Canceled: {}".format(evt))


def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print("Session Stopped: {}".format(evt))


def recognize_from_microphone():
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

    transcribing_stop = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        print("Stopping transcription due to event: {}".format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True

    # Connect callbacks to the events fired by the conversation transcriber
    conversation_transcriber.transcribed.connect(
        conversation_transcriber_transcribed_cb
    )
    conversation_transcriber.transcribing.connect(
        conversation_transcriber_transcribing_cb
    )
    conversation_transcriber.session_started.connect(
        conversation_transcriber_session_started_cb
    )
    conversation_transcriber.session_stopped.connect(
        conversation_transcriber_session_stopped_cb
    )
    conversation_transcriber.canceled.connect(
        conversation_transcriber_recognition_canceled_cb
    )
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    # Start continuous transcription
    conversation_transcriber.start_transcribing_async()

    print("Transcriber Started. Speak into the mic.\n Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping transcription...")

    conversation_transcriber.stop_transcribing_async()


# Main entry point
if __name__ == "__main__":
    try:
        recognize_from_microphone()
    except Exception as err:
        print("Encountered an exception: {}".format(err))
