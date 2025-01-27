from google.cloud import speech_v1p1beta1 as speech
import os
import json
import utils

# Define the path to the combined audio file
audio_file = "data/audio/moderate_noise.wav"

# Combine the file with signatures and generate speaker maps
speaker_maps = utils.pre_processor(audio_file)

client = speech.SpeechClient()

speech_file = ".build/combined_audio.wav"

with open(speech_file, "rb") as audio_file:
    content = audio_file.read()

audio = speech.RecognitionAudio(content=content)

diarization_config = speech.SpeakerDiarizationConfig(
    enable_speaker_diarization=True,
    min_speaker_count=2,
    max_speaker_count=10,
)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-IN",
    diarization_config=diarization_config,
)

print("Waiting for operation to complete...")
response = client.recognize(config=config, audio=audio)

# Process the diarization results
result = response.results[-1]
words_info = result.alternatives[0].words

# Group words by speaker
speaker_transcriptions = {}
for word_info in words_info:
    speaker_tag = f"Speaker {word_info.speaker_tag}"
    if speaker_tag not in speaker_transcriptions:
        speaker_transcriptions[speaker_tag] = []
    speaker_transcriptions[speaker_tag].append(word_info.word)

# Create JSON structure
final_output = []
for speaker, words in speaker_transcriptions.items():
    transcription = " ".join(words)
    final_output.append({"speaker": speaker, "transcription": transcription})

# Output as JSON
output_file = "output/gcp_diarization_output.json"
with open(output_file, "w") as f:
    json.dump(final_output, f, indent=2)

# Print the final JSON
print(json.dumps(final_output, indent=2))
