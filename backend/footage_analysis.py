#!/usr/bin/env python3
import os
import sys
import json
import shutil
import moviepy.editor as mp
import whisper
from pathlib import Path
import dotenv
import time

# Import LLM client
from llama_stack_client import LlamaStackClient

# Load environment variables
dotenv.load_dotenv()

# Constants
CHUNK_DURATION = 300  # 5 minutes in seconds
OUTPUT_FILE = "analysis_results.json"

# Connect to Llama Stack
def create_llama_client():
    return LlamaStackClient(
        base_url=f"http://llama-stack:{os.getenv('LLAMA_STACK_PORT')}"
    )

# Simplified system prompt that doesn't ask for JSON
SYSTEM_PROMPT = """
You are an AI that analyzes incidents in bodycam transcriptions.
You will be given audio transcripts from police body cameras.
If you recognize events that constitute conflict, altercations, or concerning police conduct,
summarize it and flag it for manual review. Be concise but include all relevant details.
If the transcription contains nothing of note and just normal stanard patroling, simply state that there are no incidents!
"""

# Extract audio chunks from video
def extract_chunks(video_path):
    """Split video into 5-minute chunks and extract audio as MP3"""
    print(f"Processing video: {video_path}")
    clip = mp.VideoFileClip(video_path)
    duration = int(clip.duration)
    print(f"Video duration: {duration} seconds")
    
    # Create a "temp" directory in the current working directory if it doesn't exist
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

# Transcribe audio chunks
def transcribe_chunks(audio_files, model_size='tiny'):
    """Transcribe each audio chunk using Whisper"""
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

# Analyze transcripts with LLM
def analyze_transcript(llm_client, model_id, transcript_text, chunk_start_time):
    """Send transcript to LLM for incident analysis"""
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

def main():
    if len(sys.argv) != 2:
        print("Usage: python footage_analysis.py /path/to/video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' does not exist")
        sys.exit(1)
    
    print("Connecting to Llama Stack...")
    llm_client = None
    model_id = None
    
    try:
        llm_client = create_llama_client()
        model_id = os.getenv("INFERENCE_MODEL")
        if not model_id:
            print("Error: INFERENCE_MODEL not set in .env")
            sys.exit(1)
        print(f"Using model: {model_id}")
    except Exception as e:
        print(f"Error connecting to Llama Stack: {e}")
    
    try:
        # Process the video
        audio_chunks, temp_dir, video_duration = extract_chunks(video_path)
        transcripts = transcribe_chunks(audio_files=audio_chunks)
        
        results = []
        
        for audio_path, transcript_text, start_time in transcripts:
            chunk_name = os.path.basename(audio_path)
            
            # Save transcript to temp directory
            transcript_file = os.path.join(temp_dir, f"{os.path.splitext(chunk_name)[0]}_transcript.txt")
            with open(transcript_file, 'w') as f:
                f.write(transcript_text)
            print(f"Saved transcript to: {transcript_file}")
            
            # Analyze transcript
            if llm_client and model_id:
                try:
                    incident_data = analyze_transcript(llm_client, model_id, transcript_text, start_time)
                except Exception as e:
                    print(f"LLM analysis failed! {e}")
                    incident_data = {
                        "timestamp": start_time,
                        "summary": "LLM analysis failed."
                    }
            else:
                incident_data = {
                    "timestamp": start_time,
                    "summary": "LLM client not available."
                }
            
            results.append(incident_data)
        
        # Save results to JSON
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalysis complete!")
        print(f"Total chunks processed: {len(results)}")
        print(f"Results saved to: {OUTPUT_FILE}")
        
        # Clean up all temp files
        try:
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {e}")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
