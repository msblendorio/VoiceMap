import argparse
import json
import logging
import os
import ssl
import subprocess
import sys
import threading
import time
import warnings
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import torch
import urllib.request
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import whisper
from transformers import pipeline

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
os.environ["PYANNOTE_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

FFMPEG_AVAILABLE = check_ffmpeg()
if not FFMPEG_AVAILABLE:
    raise RuntimeError("ffmpeg not found. Please install ffmpeg: brew install ffmpeg")

try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: []
except ImportError:
    pass

try:
    import pyannote.audio.core.io as pyannote_io
    import soundfile
    
    if not hasattr(pyannote_io, 'AudioDecoder') or pyannote_io.AudioDecoder is None:
        class AudioDecoder:
            def __init__(self, audio_source):
                if isinstance(audio_source, str):
                    self.audio_path = audio_source
                elif hasattr(audio_source, 'name'):
                    self.audio_path = audio_source.name
                else:
                    raise ValueError(f"Unsupported audio source type: {type(audio_source)}")
                
            @property
            def metadata(self):
                class AudioStreamMetadata:
                    pass
                try:
                    info = soundfile.info(self.audio_path)
                    metadata = AudioStreamMetadata()
                    metadata.num_channels = info.channels
                    metadata.sample_rate = info.samplerate
                    metadata.num_frames = info.frames
                    metadata.duration_seconds_from_header = info.frames / info.samplerate if info.samplerate > 0 else 0
                    return metadata
                except Exception:
                    metadata = AudioStreamMetadata()
                    metadata.num_channels = 1
                    metadata.sample_rate = 16000
                    metadata.num_frames = 0
                    metadata.duration_seconds_from_header = 0
                    return metadata
            
            def get_all_samples(self):
                import torch
                import soundfile as sf
                try:
                    waveform, sample_rate = sf.read(self.audio_path)
                    waveform_tensor = torch.from_numpy(waveform).float()
                    if len(waveform_tensor.shape) == 1:
                        waveform_tensor = waveform_tensor.unsqueeze(0)
                    
                    class AudioSamples:
                        def __init__(self, waveform, sample_rate):
                            self.waveform = waveform
                            self.data = waveform
                            self.sample_rate = sample_rate
                    
                    return AudioSamples(waveform_tensor, sample_rate)
                except Exception as e:
                    raise RuntimeError(f"Failed to load audio samples from {self.audio_path}: {e}")
            
            def get_samples_played_in_range(self, start_time, end_time):
                import torch
                import soundfile as sf
                try:
                    info = sf.info(self.audio_path)
                    sample_rate = info.samplerate
                    
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    
                    waveform, sample_rate = sf.read(self.audio_path, start=start_sample, stop=end_sample, always_2d=False)
                    
                    waveform_tensor = torch.from_numpy(waveform).float()
                    if len(waveform_tensor.shape) == 1:
                        waveform_tensor = waveform_tensor.unsqueeze(0)
                    
                    class AudioSamples:
                        def __init__(self, waveform, sample_rate):
                            self.waveform = waveform
                            self.data = waveform
                            self.sample_rate = sample_rate
                    
                    return AudioSamples(waveform_tensor, sample_rate)
                except Exception as e:
                    raise RuntimeError(f"Failed to load audio samples from {self.audio_path} (range {start_time}-{end_time}): {e}")
        
        pyannote_io.AudioDecoder = AudioDecoder
        
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation
except ImportError as e:
    print(f"Warning: Could not import pyannote modules: {e}")
    class Pipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("pyannote.audio Pipeline not available")
    class Annotation:
        def __init__(self):
            pass
        def __len__(self):
            return 0
        def labels(self):
            return set()
        def itertracks(self, yield_label=False):
            return iter([])


def setup_ssl_context():
    if os.getenv("SSL_VERIFY", "true").lower() == "false":
        return ssl._create_unverified_context()
    return ssl._create_unverified_context()

ssl_context = setup_ssl_context()

_original_urlopen = urllib.request.urlopen

def _patched_urlopen(*args, **kwargs):
    if 'context' not in kwargs:
        kwargs['context'] = ssl_context
    return _original_urlopen(*args, **kwargs)

urllib.request.urlopen = _patched_urlopen


class AudioTranscriber:
    
    def __init__(self, whisper_model: str = "large", hf_token: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}\n")
        
        with tqdm(total=3, desc="Loading models", unit="model") as pbar:
            pbar.set_description(f"Loading Whisper model '{whisper_model}'")
            start_time = time.time()
            self.whisper_model = whisper.load_model(whisper_model, device=self.device)
            pbar.update(1)
            pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
            
            if hf_token:
                pbar.set_description("Loading speaker diarization pipeline")
                start_time = time.time()
                try:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
                    if self.device == "cuda":
                        self.diarization_pipeline.to(torch.device(self.device))
                    self.diarization_enabled = True
                except Exception as e:
                    print(f"WARNING: Failed to load speaker diarization pipeline ({e}). Speaker diarization disabled.")
                    self.diarization_pipeline = None
                    self.diarization_enabled = False
                pbar.update(1)
                pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
            else:
                print("WARNING: HuggingFace token not provided. Speaker diarization disabled.")
                self.diarization_pipeline = None
                self.diarization_enabled = False
                pbar.update(1)
            
            pbar.set_description("Loading summarization model")
            start_time = time.time()
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.summarizer = pipeline(
                    "summarization",
                    model="it5/it5-efficient-small-el32-news-summarization",
                    device=0 if self.device == "cuda" else -1
                )
            pbar.update(1)
            pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
        
        print()
    
    def convert_audio(self, audio_path: Path, output_path: Path = None) -> Path:
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
    
    def transcribe(self, audio_path: Path, language: str = "it") -> Dict:
        with tqdm(desc="Transcribing with Whisper", unit="step", total=1) as pbar:
            start_time = time.time()
            result = self.whisper_model.transcribe(str(audio_path), language=language, verbose=False)
            pbar.update(1)
            pbar.set_postfix({"time": f"{time.time() - start_time:.1f}s"})
            return result
    
    def diarize(self, audio_path: Path, transcription: Dict = None, min_speakers: int = None, max_speakers: int = None) -> Annotation:
        """Perform speaker diarization using pyannote Pipeline."""
        if not self.diarization_enabled or self.diarization_pipeline is None:
            return Annotation()
        
        try:
            start_time = time.time()
            print(f"\nStarting diarization for file: {audio_path.name}")
            
            pipeline_kwargs = {}
            if min_speakers is not None:
                pipeline_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                pipeline_kwargs["max_speakers"] = max_speakers
            
            # Try to use ProgressHook if available
            try:
                from pyannote.audio.pipelines.utils.hook import ProgressHook
                with ProgressHook() as hook:
                    print("Running diarization with progress monitoring...")
                    diarization = self.diarization_pipeline(str(audio_path), hook=hook, **pipeline_kwargs)
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
                    diarization = self.diarization_pipeline(str(audio_path), **pipeline_kwargs)
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
    
    def assign_speakers_to_transcription(self, transcription: Dict, diarization: Annotation) -> List[Dict]:
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
    
    def format_transcript(self, segments: List[Dict]) -> str:
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
    
    def summarize(self, text: str, max_length: int = None) -> str:
        if max_length is None:
            max_length = int(os.getenv("SUMMARY_MAX_LENGTH", "400"))
        
        max_input = 1024
        if len(text) > max_input:
            text = text[:max_input]
        
        min_length = max(50, max_length // 4)
        
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]["summary_text"]
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Summary not available."
    
    def summarize_by_speaker(self, segments: List[Dict], max_length: int = None) -> str:
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
                                chunk_summary = self.summarize(chunk_text, max_length=max_length // 2)
                                chunk_summaries.append(chunk_summary)
                                current_chunk = [word]
                                current_length = word_length
                            else:
                                current_chunk.append(word)
                                current_length += word_length
                        
                        if current_chunk:
                            chunk_text = " ".join(current_chunk)
                            chunk_summary = self.summarize(chunk_text, max_length=max_length // 2)
                            chunk_summaries.append(chunk_summary)
                        
                        combined_chunks = " ".join(chunk_summaries)
                        if len(combined_chunks) > max_input:
                            summary_text = self.summarize(combined_chunks, max_length=max_length)
                        else:
                            summary_text = combined_chunks
                    else:
                        summary_text = self.summarize(speaker_content, max_length=max_length)
                    
                    summaries.append(f"{speaker}: {summary_text}")
                except Exception as e:
                    summaries.append(f"{speaker}: Errore durante la sintesi ({e})")
                
                pbar.update(1)
        
        return "\n\n".join(summaries)
    
    def process_audio(self, audio_path: str, output_dir: str = "output", language: str = "it") -> Dict:
        total_start_time = time.time()
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        steps = [
            ("convert_audio", "Audio conversion"),
            ("transcribe", "Transcription"),
            ("diarize", "Speaker diarization"),
            ("assign_speakers", "Assigning speakers"),
            ("format_transcript", "Formatting transcript"),
            ("summarize", "Summarization"),
            ("save_files", "Saving results")
        ]
        
        with tqdm(total=len(steps), desc="Processing audio", unit="step") as main_pbar:
            step_times = {}
            
            main_pbar.set_description("Converting audio")
            step_start = time.time()
            wav_path = self.convert_audio(audio_path, output_dir / "audio_converted.wav")
            step_times["convert_audio"] = time.time() - step_start
            main_pbar.update(1)
            
            main_pbar.set_description("Transcribing")
            step_start = time.time()
            transcription = self.transcribe(wav_path, language=language)
            step_times["transcribe"] = time.time() - step_start
            main_pbar.update(1)
            
            main_pbar.set_description("Running diarization")
            step_start = time.time()
            min_speakers_str = os.getenv("MIN_SPEAKERS")
            min_speakers = int(min_speakers_str) if min_speakers_str else None
            max_speakers_str = os.getenv("MAX_SPEAKERS")
            max_speakers = int(max_speakers_str) if max_speakers_str else None
            diarization = self.diarize(wav_path, transcription, min_speakers=min_speakers, max_speakers=max_speakers)
            step_times["diarize"] = time.time() - step_start
            main_pbar.update(1)
            
            main_pbar.set_description("Assigning speakers")
            step_start = time.time()
            segments = self.assign_speakers_to_transcription(transcription, diarization)
            step_times["assign_speakers"] = time.time() - step_start
            main_pbar.update(1)
            
            main_pbar.set_description("Formatting transcript")
            step_start = time.time()
            formatted_transcript = self.format_transcript(segments)
            step_times["format_transcript"] = time.time() - step_start
            main_pbar.update(1)
            
            main_pbar.set_description("Generating summary")
            step_start = time.time()
            summary = self.summarize_by_speaker(segments)
            step_times["summarize"] = time.time() - step_start
            main_pbar.update(1)
            
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
                import re
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
    
    transcriber = AudioTranscriber(whisper_model=args.whisper_model, hf_token=hf_token, device=args.device)
    transcriber.process_audio(audio_path=args.audio_file, output_dir=args.output_dir, language=args.language)
    
    print('Process completed successfully and files saved in output directory: ' + args.output_dir)

if __name__ == "__main__":
    main()
