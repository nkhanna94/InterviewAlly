import os
from typing import Dict, Any, List
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch

try:
    from torch.torch_version import TorchVersion
    torch.serialization.add_safe_globals([TorchVersion])
except Exception as e:
    print(f"Warning: Could not whitelist TorchVersion: {e}")
# ----------------------------------------

# Configuration
MODEL_SIZE = "base.en"

if torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
    DIARIZATION_DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
elif torch.backends.mps.is_available():

    WHISPER_DEVICE = "cpu"    # CTranslate2 crashes on MPS, force CPU
    DIARIZATION_DEVICE = "mps" # Pyannote runs great on MPS
    COMPUTE_TYPE = "int8"     # int8 is faster on CPU
else:
    WHISPER_DEVICE = "cpu"
    DIARIZATION_DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

HF_TOKEN = os.getenv("HF_TOKEN") 

class TranscriptGenerator:
    def __init__(self):
        print(f"Loading Whisper model: {MODEL_SIZE} on {WHISPER_DEVICE} (Compute: {COMPUTE_TYPE})...")
        self.model = WhisperModel(MODEL_SIZE, device=WHISPER_DEVICE, compute_type=COMPUTE_TYPE)
        
        # Initialize Diarization Pipeline
        if HF_TOKEN:
            print(f"Loading Diarization pipeline on {DIARIZATION_DEVICE}...")
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_TOKEN
                ).to(torch.device(DIARIZATION_DEVICE))
            except Exception as e:
                print(f"Error loading Pyannote: {e}")
                self.diarization_pipeline = None
        else:
            print("⚠️ No HF_TOKEN found. Diarization will be skipped.")
            self.diarization_pipeline = None

    def process_video_file(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            print(f"Transcribing {file_path}...")
            
            # 1. Transcribe (Get Text + Timestamps)
            segments, info = self.model.transcribe(file_path, beam_size=5)
            whisper_segments = list(segments) 
            
            final_transcript = []
            
            # 2. Diarization (Get Speakers + Timestamps)
            if self.diarization_pipeline:
                print("Running Speaker Diarization...")
                # Pyannote expects a path or waveform. Passing path is easiest.
                diarization = self.diarization_pipeline(file_path)
                
                # 3. Merge Logic: Assign Speaker to Text
                for seg in whisper_segments:
                    mid_time = (seg.start + seg.end) / 2
                    speaker_label = "Unknown" 
                    
                    # Find speaker active at the midpoint of the word/segment
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        if turn.start <= mid_time <= turn.end:
                            speaker_label = f"Speaker {speaker}"
                            break
                    
                    final_transcript.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                        "speaker": speaker_label
                    })
            else:
                # Fallback
                for seg in whisper_segments:
                    final_transcript.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip(),
                        "speaker": "Speaker"
                    })

            return final_transcript
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise e

def create_transcript_generator():
    return TranscriptGenerator()