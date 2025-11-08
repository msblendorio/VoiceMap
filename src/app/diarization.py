import sys
import threading
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from utils import Pipeline, Annotation


def load_diarization_pipeline(hf_token: str, device: str = None) -> Tuple[Optional[Pipeline], bool]:
    """Load speaker diarization pipeline."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not hf_token:
        print("WARNING: HuggingFace token not provided. Speaker diarization disabled.")
        return None, False
    
    with tqdm(desc="Loading speaker diarization pipeline", unit="model", total=1) as pbar:
        start_time = time.time()
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            if device == "cuda":
                pipeline.to(torch.device(device))
            pbar.update(1)
            pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
            return pipeline, True
        except Exception as e:
            print(f"WARNING: Failed to load speaker diarization pipeline ({e}). Speaker diarization disabled.")
            pbar.update(1)
            pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
            return None, False


def diarize(audio_path: Path, diarization_pipeline: Optional[Pipeline], 
            min_speakers: int = None, max_speakers: int = None) -> Annotation:
    """Perform speaker diarization using pyannote Pipeline."""
    if diarization_pipeline is None:
        return Annotation()
    
    try:
        start_time = time.time()
        print(f"\nStarting diarization for file: {audio_path.name}")
        
        pipeline_kwargs = {}
        if min_speakers is not None:
            pipeline_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            pipeline_kwargs["max_speakers"] = max_speakers

        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            with ProgressHook() as hook:
                print("Running diarization with progress monitoring...")
                diarization = diarization_pipeline(str(audio_path), hook=hook, **pipeline_kwargs)
        except ImportError:
            def show_elapsed():
                while not diarization_done.is_set():
                    elapsed = time.time() - start_time
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)
                    sys.stdout.write(f"\r[Diarization in progress... Elapsed time: {minutes}m {seconds}s]")
                    sys.stdout.flush()
                    if diarization_done.wait(2):
                        break
            
            diarization_done = threading.Event()
            timer_thread = threading.Thread(target=show_elapsed, daemon=True)
            timer_thread.start()
            
            print("Running diarization...")
            try:
                diarization = diarization_pipeline(str(audio_path), **pipeline_kwargs)
            finally:
                diarization_done.set()
                timer_thread.join(timeout=1)
                print()
        
        elapsed = time.time() - start_time
        detected_speakers = len(diarization.labels())
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"\nDiarization completed in {minutes}m {seconds}s: {len(diarization)} segments, {detected_speakers} speakers found")
        return diarization
    except Exception as e:
        print(f"\nWarning: Speaker diarization failed ({type(e).__name__}: {e}). Continuing without diarization.")
        import traceback
        traceback.print_exc()
        return Annotation()


def assign_speakers_to_transcription(transcription: Dict, diarization: Annotation) -> List[Dict]:
    """Assign speakers to transcription segments by matching timestamps."""
    segments = transcription["segments"]
    
    if not diarization or len(diarization) == 0:
        return [
            {"start": round(seg["start"], 2), "end": round(seg["end"], 2), "text": seg["text"], "speaker": "SPEAKER_UNKNOWN"}
            for seg in segments
        ]
    
    print("\nAssigning speakers to transcribed segments...")
    
    segments_with_speakers = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_mid = (seg_start + seg_end) / 2
        
        speaker = "SPEAKER_UNKNOWN"
        best_overlap = 0
        
        for segment, track, label in diarization.itertracks(yield_label=True):
            overlap_start = max(seg_start, segment.start)
            overlap_end = min(seg_end, segment.end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > best_overlap:
                    best_overlap = overlap_duration
                    speaker = label
        
        if speaker == "SPEAKER_UNKNOWN":
            for segment, track, label in diarization.itertracks(yield_label=True):
                if segment.start <= seg_mid <= segment.end:
                    speaker = label
                    break
        
        segments_with_speakers.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"],
            "speaker": speaker
        })
    
    return segments_with_speakers
