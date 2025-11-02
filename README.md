# audio-transcription-diarization

Python pipeline for audio transcription with speaker diarization and summarization in Italian. Uses Whisper for transcription and pyannote.audio for speaker recognition.

## 📋 Requirements

- Python 3.12
- HuggingFace Token (required only for speaker diarization)

## 🔑 Getting the HuggingFace Token

The HuggingFace token is required to use the `pyannote/speaker-diarization-3.1` model. Here's how to get it:

### Step 1: Sign up on HuggingFace
1. Go to [huggingface.co](https://huggingface.co)
2. Create an account or log in

### Step 2: Create a Token
1. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Assign a name (e.g., "audio-transcription")
4. Select the type: **Read** (read-only)
5. Click **"Generate token"**
6. **Copy the token** (it will only be visible this once)

### Step 3: Accept Model Terms
1. Go to the model page: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Look for an **"Accept"** button or a terms acceptance section on the model page
3. Click **"Accept"** to accept the terms of use
   
   **Note**: If you don't see an "Accept" button:
   - The terms may already be accepted with your account
   - Try accessing the model files or check for any banners/notifications on the page
   - The acceptance prompt may appear when you first try to use the model with your token

### Step 4: Configure the Token

**Option A: Environment variable (recommended)**

```bash
export HF_TOKEN="hf_yourtokenhere"
```

**Option B: .env file (local development)**

Create a `.env` file in the project root:
```
HF_TOKEN=hf_yourtokenhere
```

⚠️ **Note**: The script works without a token, but speaker diarization will be disabled. Transcription and summarization will still work.

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 💻 Usage

```bash
python main.py <audio_file> [options]
```

### Examples

**Basic usage (with token from environment variable):**
```bash
python main.py audio.mp3
```

**Specify Whisper model:**
```bash
python main.py audio.mp3 --whisper-model large
```

**Specify token via CLI (override):**
```bash
python main.py audio.mp3 --hf-token "hf_yourtokenhere"
```

**Specify output directory:**
```bash
python main.py audio.mp3 --output-dir results
```

**Specify transcription language:**
```bash
python main.py audio.mp3 --language en
python main.py audio.mp3 --language es
python main.py audio.mp3 --language fr
```

**Specify number of speakers (optional):**
```bash
python main.py audio.mp3 --min-speakers 2 --max-speakers 4
```

**Complete example with all options:**
```bash
python main.py "data/audio_file.mpeg" --whisper-model large-v3-turbo --language it --output-dir output --min-speakers 2 --max-speakers 5
```

### Available options

- `--whisper-model`: Whisper model name (`tiny`, `base`, `small`, `medium`, `large`, `large-v3-turbo`) - Default: `large-v3-turbo`
- `--language`: Language code for transcription (`it`, `en`, `es`, `fr`, `de`, etc.) - Default: `it` (Italian)
- `--hf-token`: HuggingFace token (optional if already configured as env var)
- `--output-dir`: Output directory for files - Default: `output`
- `--device`: Device to use (`cuda`, `cpu`) - Default: auto-detect
- `--min-speakers`: Minimum number of speakers (optional, uses MIN_SPEAKERS env var if not specified)
- `--max-speakers`: Maximum number of speakers (optional, uses MAX_SPEAKERS env var if not specified)

## 📁 Output

The script generates in the output directory:

- `transcript.txt`: Formatted transcription with speakers
- `summary.txt`: Summary of the transcribed text
- `results.json`: Complete JSON with all data
- `audio_converted.wav`: Audio file converted to WAV 16kHz mono
