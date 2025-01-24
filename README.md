# Speaker Diarization and Recognition
This repository contains the source code and data for efficient speaker recognition, transcription and diarization.
The `doc` folder contains documentation for different diarization pipelines.
### Requirements
Two environment variables: `SPEECH_KEY` and `SPEECH_REGION` are necessary for the Azure API used in the source code to function.
### Source Code
1. `real-time-diarize.py`
    This file performs diarization in real time using the default microphone on the device. Azure API cannot use speech signatures in real-time to verify the identity of any speaker; different speakers are differentiated dynamically and are assigned unique IDs.
2. `meeting_transciption.py`
    Gives the speaker diarized transcript of a meeting using voice signatures recorded earlier. Voice signatures can only be recorded using Azure's REST API.
3. `real-time-verify-diarize.py`
    Pushes small signature voice clips into the Azure API before real-time diarization is performed so that we can map the dynamic IDs assigned by Azure with the actual names of the speakers.
### Data
1. `signatures`
    Contains speaker voice signatures
2. `speaker_id.json`
    Contains a mapping between speaker voice signatures in `signatures` and their names.
