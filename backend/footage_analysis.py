#!/usr/bin/env python3
import os
import json
import shutil
import moviepy.editor as mp
import whisper
from pathlib import Path
import dotenv
import time
import sys

# Import LLM client
from llama_stack_client import LlamaStackClient

# Load environment variables
dotenv.load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

# Constants
CHUNK_DURATION = 300  # 5 minutes in seconds
SYSTEM_PROMPT = """
You are an AI that analyzes incidents in bodycam transcriptions.
You will be given audio transcripts from police body cameras.
If you recognize events that constitute conflict, altercations, or concerning police conduct,
summarize it and flag it for manual review. Be concise but include all relevant details.
If the transcription contains nothing of note and just normal stanard patroling, simply state that there are no incidents!
"""

def create_llama_client():
    return LlamaStackClient(
        base_url=f"http://llama-stack:{os.getenv('LLAMA_STACK_PORT')}"
    )

def extract_chunks(video_path):
    print(f"Processing video: {video_path}")
    clip = mp.VideoFileClip(video_path)
    duration = int(clip.duration)
    print(f"Video duration: {duration} seconds")

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    print(f"Using temporary directory: {temp_dir.resolve()}")

    audio_files = []

    for start in range(0, duration, CHUNK_DURATION):
        end = min(start + CHUNK_DURATION, duration)
        print(f"Extracting chunk {start//60}-{end//60} minutes...")
        subclip = clip.subclip(start, end)
        audio_filename = temp_dir / f"chunk_{start // 60:02d}-{end // 60:02d}.mp3"
        subclip.audio.write_audiofile(str(audio_filename), codec='libmp3lame', verbose=False, logger=None)
        audio_files.append((str(audio_filename), start))
        print(f"Saved chunk: {audio_filename} (starts at {start}s)")

    return audio_files, str(temp_dir), duration

def transcribe_chunks(audio_files, model_size='tiny'):
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    transcripts = []

    for audio_path, start_time in audio_files:
        print(f"Transcribing: {audio_path}")
        start = time.time()
        result = model.transcribe(audio_path)
        end = time.time()
        print(f"Transcription completed in {end-start:.2f} seconds")
        transcripts.append((audio_path, result['text'], start_time))

    return transcripts

def analyze_transcript(llm_client, model_id, transcript_text, chunk_start_time):
    print(f"Analyzing transcript for chunk starting at {chunk_start_time}s")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{transcript_text}"}
    ]

    try:
        response = llm_client.inference.chat_completion(
            model_id=model_id,
            messages=messages
        )

        summary = response.completion_message.content
        print(f"LLM response received: {summary[:100]}...")

        has_incident = "no incident" not in summary.lower() and "nothing of note" not in summary.lower()

        return {
            "timestamp": chunk_start_time,
            "summary": summary
        }

    except Exception as e:
        print(f"Error analyzing transcript: {e}")
        return {
            "timestamp": chunk_start_time,
            "summary": f"Error analyzing transcript: {str(e)}"
        }

def analyze_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File '{video_path}' does not exist")

    print("Connecting to Llama Stack...")
    llm_client = create_llama_client()
    model_id = os.getenv("INFERENCE_MODEL")
    if not model_id:
        raise RuntimeError("INFERENCE_MODEL not set in .env")
    print(f"Using model: {model_id}")

    audio_chunks, temp_dir, video_duration = extract_chunks(video_path)
    transcripts = transcribe_chunks(audio_chunks)

    results = []

    for audio_path, transcript_text, start_time in transcripts:
        chunk_name = os.path.basename(audio_path)

        # Save transcript (optional, useful for debugging)
        transcript_file = os.path.join(temp_dir, f"{os.path.splitext(chunk_name)[0]}_transcript.txt")
        with open(transcript_file, 'w') as f:
            f.write(transcript_text)

        try:
            incident_data = analyze_transcript(llm_client, model_id, transcript_text, start_time)
        except Exception as e:
            incident_data = {
                "timestamp": start_time,
                "summary": f"LLM analysis failed: {str(e)}"
            }

        chunk_end_time = min(start_time + CHUNK_DURATION, video_duration)

        incident_data.update({
            "start_time_seconds": start_time,
            "end_time_seconds": chunk_end_time,
            "start_time_minsec": f"{start_time // 60:02d}:{start_time % 60:02d}",
            "end_time_minsec": f"{chunk_end_time // 60:02d}:{chunk_end_time % 60:02d}"
        })

        results.append(incident_data)

    # Clean up
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory: {e}")

    return results

# CLI usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python footage_analysis.py /path/to/video.mp4")
        exit(1)

    path = sys.argv[1]
    results = analyze_video(path)

    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Analysis complete. Results saved to analysis_results.json.")
