import os
import requests
import json
import utils

# Define the path to the combined audio file
audio_file = "data/audio/big_crowd.wav"

# Get environment variables
SPEECH_REGION = os.getenv("SPEECH_REGION")
SPEECH_KEY = os.getenv("SPEECH_KEY")

# Define the URL
url = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"

# Combine the file with signatures and generate speaker maps
speaker_maps = utils.pre_processor(audio_file)

# The path to the combined audio file
audio_file_path = ".build/combined_final.wav"

definition = {
    "locales": ["en-IN", "en-US"],
    "diarization": {"maxSpeakers": 7, "enabled": True},
}

# Open the audio file and prepare the request
with open(audio_file_path, "rb") as audio_file:
    files = {
        "audio": audio_file,  # The audio file in binary mode
        "definition": (
            None,
            json.dumps(definition),
            "application/json",
        ),  # Definition as JSON with content type
    }

    # Send the POST request
    response = requests.post(
        url, headers={"Ocp-Apim-Subscription-Key": SPEECH_KEY}, files=files
    )

# Handle the response
if response.status_code == 200:
    for i in utils.parse_speaker_text(response.json(), speaker_maps):
        print(i, end="\n\n")
else:
    print(f"Request failed with status code {response.status_code}")
    print(utils.parse_speaker_text(response.text))
