import subprocess
import time
import whisper
import torch
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from utils import PYDUB_AVAILABLE, check_ffmpeg

if PYDUB_AVAILABLE:
    from pydub import AudioSegment

FFMPEG_AVAILABLE = check_ffmpeg()
if not FFMPEG_AVAILABLE:
    raise RuntimeError("ffmpeg not found. Please install ffmpeg: brew install ffmpeg")


def load_whisper_model(model_name: str, device: str = None) -> whisper.Whisper:
    """Load Whisper model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with tqdm(desc=f"Loading Whisper model '{model_name}'", unit="model", total=1) as pbar:
        start_time = time.time()
        model = whisper.load_model(model_name, device=device)
        pbar.update(1)
        pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
    
    return model


def convert_audio(audio_path: Path, output_path: Path = None) -> Path:
    """Convert audio file to WAV format (16kHz, mono)."""
    if output_path is None:
        output_path = audio_path.with_suffix('.wav')
    
    with tqdm(desc="Converting audio", unit="step", total=1) as pbar:
        if PYDUB_AVAILABLE:
            try:
                start_time = time.time()
                audio = AudioSegment.from_file(str(audio_path))
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(str(output_path), format="wav")
                pbar.update(1)
                pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
                return output_path
            except Exception as e:
                pbar.set_description(f"pydub failed, using ffmpeg... ({e})")
        
        try:
            start_time = time.time()
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path), "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-y", str(output_path)],
                check=True,
                capture_output=True
            )
            pbar.update(1)
            pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio conversion failed with ffmpeg: {e.stderr.decode() if e.stderr else str(e)}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg or ensure pydub works properly.")


def transcribe(audio_path: Path, whisper_model: whisper.Whisper, language: str = "it") -> Dict:
    """Transcribe audio file using Whisper."""
    with tqdm(desc="Transcribing with Whisper", unit="step", total=1) as pbar:
        start_time = time.time()
        result = whisper_model.transcribe(str(audio_path), language=language, verbose=False)
        pbar.update(1)
        pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
        return result
