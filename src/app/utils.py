import logging
import os
import ssl
import subprocess
import urllib.request
import warnings

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


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


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
