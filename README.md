# Speaker Diarization and Recognition
This repository contains the source code and data for efficient speaker recognition, transcription and diarization.
The `doc` folder contains documentation for different diarization pipelines.
### Requirements
Two environment variables: `SPEECH_KEY` and `SPEECH_REGION` are necessary for the Azure API used in the source code to function.

`ffmpeg` is required for efficient audio format conversion:
```python
sudo apt install ffmpeg  # For Linux
brew install ffmpeg      # For macOS
```

Requirements can be installed in a separate conda environment using `environment.yml`:
```python
conda create -f environment.yml
```
### Source Code
1. `real-time-diarize.py`
    This file performs diarization in real time using the default microphone on the device. Azure API cannot use speech signatures in real-time to verify the identity of any speaker; different speakers are differentiated dynamically and are assigned unique IDs.
2. `azure_diarization.py`
    Performs diarization really fast with pre-recorded signatures and an audio file. Best performing API.
3. `gcp_diarization.py`
    Uses Google's Speech-to-Text API for transcription and diarization. Very unsatisfactory results.
### Data
1. `signatures`
    Contains speaker voice signatures
2. `audio`
    Contains the audio files for testing.
3. `.bulid`
    Excluded in the repo; used for storing combined audio files before processing.
### Output
Contains diarization output files, used for debugging.
