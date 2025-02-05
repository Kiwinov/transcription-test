import os
import requests
import json
import utils
import time
from dotenv import load_dotenv
import asyncio

# Define the path to the audio file needed to be transcribed
audio_file = "audio/wazeer_1738076770.wav"

# Define the definition for the transcription
definition = {
    "locales": ["en-IN", "en-US"],
    "diarization": {"maxSpeakers": 12, "enabled": True},
    "profanityFilterMode": "None",
}


async def setup_azure():
    """
    Set up the Azure Speech service.
    """
    # Load the environment variables
    load_dotenv()
    # Get environment variables
    SPEECH_REGION = os.getenv("SPEECH_REGION")
    SPEECH_KEY = os.getenv("SPEECH_KEY")

    # Define the URL
    url = f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"

    return url, SPEECH_KEY


async def transcribe_azure(definition, url, SPEECH_KEY):
    """
    Transcribe the audio file using Azure Speech service.

    Args:
        definition (dict): The definition for the transcription.
        url (str): The URL for the Azure Speech service.

    Returns:
        None
    """
    # Combine the file with signatures and generate speaker maps
    speaker_maps, final_output_path = await utils.speaker_map_processor(audio_file)

    # Open the audio file and prepare the request
    with open(final_output_path, "rb") as file:
        files = {
            "audio": file,  # The audio file in binary mode
            "definition": (
                None,
                json.dumps(definition),
                "application/json",
            ),  # Definition as JSON with content type
        }

        t1 = time.time()
        # Send the POST request
        response = requests.post(
            url, headers={"Ocp-Apim-Subscription-Key": SPEECH_KEY}, files=files
        )
        t2 = time.time()
        print(f"Time taken: {t2 - t1} seconds")

    # Handle the response
    if response.status_code == 200:
        output_dir = "output_log"
        os.makedirs(output_dir, exist_ok=True)
        with open(
            f"output/azure_diarization_output_{audio_file.split("/")[-1].split(".")[0]}.txt",
            "w",
        ) as f:
            for i in await utils.parse_speaker_text(response.json(), speaker_maps):
                if (
                    float(i.split("Offset:")[-1].split("Duration:")[0])
                    > await utils.get_wav_duration(".build/combined_signs.wav") - 1
                ):
                    print(i, end="\n\n")
                    f.write(i + "\n\n")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(utils.parse_speaker_text(response.text))


url, SPEECH_KEY = asyncio.run(setup_azure())
asyncio.run(transcribe_azure(definition, url, SPEECH_KEY))
