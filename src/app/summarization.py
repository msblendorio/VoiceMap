import os
import time
import torch
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from typing import Dict, List
from tqdm import tqdm
from transformers import pipeline


def load_summarizer(device: str = None) -> pipeline:
    """Load summarization model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with tqdm(desc="Loading summarization model", unit="model", total=1) as pbar:
        start_time = time.time()
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            summarizer = pipeline(
                "summarization",
                model="it5/it5-efficient-small-el32-news-summarization",
                device=0 if device == "cuda" else -1
            )
        pbar.update(1)
        pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
    
    return summarizer


def summarize(text: str, summarizer: pipeline, max_length: int = None) -> str:
    """Summarize text using the summarization model."""
    if max_length is None:
        max_length = int(os.getenv("SUMMARY_MAX_LENGTH", "400"))
    
    max_input = 1024
    if len(text) > max_input:
        text = text[:max_input]
    
    min_length = max(50, max_length // 4)
    
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Summary not available."


def summarize_by_speaker(segments: List[Dict], summarizer: pipeline, max_length: int = None) -> str:
    """Group segments by speaker and summarize each speaker's contributions."""
    if max_length is None:
        max_length = int(os.getenv("SUMMARY_MAX_LENGTH", "400"))
    
    speaker_texts = {}
    
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
        text = seg.get("text", "").strip()
        if text:
            if speaker not in speaker_texts:
                speaker_texts[speaker] = []
            speaker_texts[speaker].append(text)
    
    if not speaker_texts:
        return "No content to summarize."
    
    summaries = []
    total_speakers = len(speaker_texts)
    max_input = 1024
    
    with tqdm(desc="Generating summary by speaker", total=total_speakers, unit="speaker") as pbar:
        for speaker, texts in sorted(speaker_texts.items()):
            speaker_content = " ".join(texts)
            
            try:
                if len(speaker_content) > max_input:
                    chunk_summaries = []
                    words = speaker_content.split()
                    current_chunk = []
                    current_length = 0
                    
                    for word in words:
                        word_length = len(word) + 1
                        if current_length + word_length > max_input and current_chunk:
                            chunk_text = " ".join(current_chunk)
                            chunk_summary = summarize(chunk_text, summarizer, max_length=max_length // 2)
                            chunk_summaries.append(chunk_summary)
                            current_chunk = [word]
                            current_length = word_length
                        else:
                            current_chunk.append(word)
                            current_length += word_length
                    
                    if current_chunk:
                        chunk_text = " ".join(current_chunk)
                        chunk_summary = summarize(chunk_text, summarizer, max_length=max_length // 2)
                        chunk_summaries.append(chunk_summary)
                    
                    combined_chunks = " ".join(chunk_summaries)
                    if len(combined_chunks) > max_input:
                        summary_text = summarize(combined_chunks, summarizer, max_length=max_length)
                    else:
                        summary_text = combined_chunks
                else:
                    summary_text = summarize(speaker_content, summarizer, max_length=max_length)
                
                summaries.append(f"{speaker}: {summary_text}")
            except Exception as e:
                summaries.append(f"{speaker}: Errore durante la sintesi ({e})")
            
            pbar.update(1)
    
    return "\n\n".join(summaries)


def format_transcript(segments: List[Dict]) -> str:
    """Format transcript segments into readable text."""
    formatted = []
    current_speaker = None
    current_text = []
    last_break_time = 0
    
    for seg in segments:
        if seg["speaker"] != current_speaker:
            if current_text:
                formatted.append(f"{current_speaker}: {' '.join(current_text)}")
                formatted.append("")
            current_speaker = seg["speaker"]
            current_text = [seg["text"].strip()]
            last_break_time = seg["end"]
        else:
            if seg["start"] - last_break_time > 30:
                if current_text:
                    formatted.append(f"{current_speaker}: {' '.join(current_text)}")
                    formatted.append("")
                current_text = [seg["text"].strip()]
                last_break_time = seg["end"]
            else:
                current_text.append(seg["text"].strip())
    
    if current_text:
        formatted.append(f"{current_speaker}: {' '.join(current_text)}")
    
    return "\n".join(formatted)
