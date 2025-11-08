import argparse
import json
import os
import re
import time
import torch
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from transcription import load_whisper_model, convert_audio, transcribe
from diarization import load_diarization_pipeline, diarize, assign_speakers_to_transcription
from summarization import load_summarizer, summarize_by_speaker, format_transcript


def process_audio(audio_path: str, output_dir: str = "output", language: str = "it",
                  whisper_model_name: str = "large-v3-turbo", hf_token: str = None,
                  device: str = None) -> Dict:
    """Process audio file: transcribe, diarize, and summarize."""
    total_start_time = time.time()
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}\n")
    
    steps = [
        ("convert_audio", "Audio conversion"),
        ("load_models", "Loading models"),
        ("transcribe", "Transcription"),
        ("diarize", "Speaker diarization"),
        ("assign_speakers", "Assigning speakers"),
        ("format_transcript", "Formatting transcript"),
        ("summarize", "Summarization"),
        ("save_files", "Saving results")
    ]
    
    with tqdm(total=len(steps), desc="Processing audio", unit="step") as main_pbar:
        step_times = {}
        
        # Convert audio
        main_pbar.set_description("Converting audio")
        step_start = time.time()
        wav_path = convert_audio(audio_path, output_dir / "audio_converted.wav")
        step_times["convert_audio"] = time.time() - step_start
        main_pbar.update(1)
        
        # Load models
        main_pbar.set_description("Loading models")
        step_start = time.time()
        whisper_model = load_whisper_model(whisper_model_name, device=device)
        diarization_pipeline, diarization_enabled = load_diarization_pipeline(hf_token, device=device)
        summarizer = load_summarizer(device=device)
        print()
        step_times["load_models"] = time.time() - step_start
        main_pbar.update(1)
        
        # Transcribe
        main_pbar.set_description("Transcribing")
        step_start = time.time()
        transcription = transcribe(wav_path, whisper_model, language=language)
        step_times["transcribe"] = time.time() - step_start
        main_pbar.update(1)
        
        # Diarize
        main_pbar.set_description("Running diarization")
        step_start = time.time()
        min_speakers_str = os.getenv("MIN_SPEAKERS")
        min_speakers = int(min_speakers_str) if min_speakers_str else None
        max_speakers_str = os.getenv("MAX_SPEAKERS")
        max_speakers = int(max_speakers_str) if max_speakers_str else None
        diarization_result = diarize(wav_path, diarization_pipeline, min_speakers=min_speakers, max_speakers=max_speakers)
        step_times["diarize"] = time.time() - step_start
        main_pbar.update(1)
        
        # Assign speakers
        main_pbar.set_description("Assigning speakers")
        step_start = time.time()
        segments = assign_speakers_to_transcription(transcription, diarization_result)
        step_times["assign_speakers"] = time.time() - step_start
        main_pbar.update(1)
        
        # Format transcript
        main_pbar.set_description("Formatting transcript")
        step_start = time.time()
        formatted_transcript = format_transcript(segments)
        step_times["format_transcript"] = time.time() - step_start
        main_pbar.update(1)
        
        # Summarize
        main_pbar.set_description("Generating summary")
        step_start = time.time()
        summary = summarize_by_speaker(segments, summarizer)
        step_times["summarize"] = time.time() - step_start
        main_pbar.update(1)
        
        # Save results
        full_text = transcription["text"]
        results = {
            "transcript": formatted_transcript,
            "summary": summary,
            "segments": segments,
            "full_text": full_text
        }
        
        main_pbar.set_description("Saving results")
        step_start = time.time()
        transcript_path = output_dir / "transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(formatted_transcript)
        
        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        json_path = output_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json_str = json.dumps(results, ensure_ascii=False, indent=2)
            json_str = re.sub(r'\b(\d+\.\d+)\b', lambda m: f'{float(m.group(1)):.2f}', json_str)
            f.write(json_str)
        
        step_times["save_files"] = time.time() - step_start
        main_pbar.update(1)
        
        total_time = time.time() - total_start_time
        main_pbar.set_postfix({"total_time": f"{total_time:.1f}s"})
        
        print(f"\nProcessing completed in {total_time:.1f}s")
        print("Step timings:")
        for step, step_time in step_times.items():
            print(f"  {step}: {step_time:.1f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="VoiceMap: Audio transcription with speaker diarization and AI-powered summarization")
    parser.add_argument("audio_file", type=str, help="Path to the MPEG audio file to process")
    parser.add_argument("--whisper-model", type=str, default="large-v3-turbo", help="Whisper model name (e.g., tiny, base, small, medium, large, large-v3-turbo). Default: large-v3-turbo")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for pyannote (optional, uses HF_TOKEN env var if not specified)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save results (default: output)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use (auto-detect if not specified)")
    parser.add_argument("--language", type=str, default="it", help="Language code for transcription (e.g., it, en, es, fr, de). Default: it (Italian)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers (uses MIN_SPEAKERS env var if not specified)")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers (uses MAX_SPEAKERS env var if not specified)")
    parser.add_argument("--summary-max-length", type=int, default=None, help="Maximum length for summary per speaker (uses SUMMARY_MAX_LENGTH env var if not specified, default: 400)")
    
    args = parser.parse_args()
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    if args.min_speakers is not None:
        os.environ["MIN_SPEAKERS"] = str(args.min_speakers)
    if args.max_speakers is not None:
        os.environ["MAX_SPEAKERS"] = str(args.max_speakers)
    if args.summary_max_length is not None:
        os.environ["SUMMARY_MAX_LENGTH"] = str(args.summary_max_length)
    
    process_audio(
        audio_path=args.audio_file,
        output_dir=args.output_dir,
        language=args.language,
        whisper_model_name=args.whisper_model,
        hf_token=hf_token,
        device=args.device
    )
    
    print(f'Process completed successfully and files saved in the directory {args.output_dir}')


if __name__ == "__main__":
    main()
