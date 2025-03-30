import os
import moviepy.editor as mp
import whisper
from pathlib import Path

# Constants
CHUNK_DURATION = 300  # 5 minutes in seconds

# Step 1: Split video into 5-minute chunks and save as MP3
def extract_chunks(video_path, output_dir):
    clip = mp.VideoFileClip(video_path)
    duration = int(clip.duration)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    audio_files = []
    for start in range(0, duration, CHUNK_DURATION):
        end = min(start + CHUNK_DURATION, duration)
        subclip = clip.subclip(start, end)
        audio_filename = os.path.join(output_dir, f"chunk_{start // 60:02d}-{end // 60:02d}.mp3")
        subclip.audio.write_audiofile(audio_filename, codec='libmp3lame', verbose=False, logger=None)
        audio_files.append(audio_filename)
        print(f"Saved chunk: {audio_filename}")
    
    return audio_files

# Step 2: Transcribe each MP3 chunk
def transcribe_chunks(audio_files, model_size='base'):
    model = whisper.load_model(model_size)
    transcripts = []
    
    for audio_path in audio_files:
        print(f"Transcribing: {audio_path}")
        result = model.transcribe(audio_path)
        transcripts.append((audio_path, result['text']))
    
    return transcripts

# Step 3: Save the full transcription
def save_transcription(transcripts, output_file="full_transcription.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for audio_path, text in transcripts:
            f.write(f"\n=== {audio_path} ===\n{text}\n")
    print(f"Transcription saved to {output_file}")

# --- Example usage ---
video_path = "test_data/boulder_police_harassment.mp4"
output_dir = "audio_chunks"

audio_files = extract_chunks(video_path, output_dir)
transcripts = transcribe_chunks(audio_files, model_size='base')
save_transcription(transcripts)
