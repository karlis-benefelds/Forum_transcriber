import whisper
import torch
import json
import re
from datetime import timedelta
from pathlib import Path
from pydub import AudioSegment
from tqdm.auto import tqdm
from contextlib import nullcontext
from typing import Optional, Callable
from src.utils import get_temp_path
from src.model_manager import ModelManager

def normalize_sentence_spacing(text: str) -> str:
    """Fix glued sentences & punctuation spacing (respects ellipses), collapse newlines/spaces."""
    if not text:
        return text
    # remove zero-widths & non-breaking spaces, collapse newlines
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = text.replace('\u00A0', ' ')
    text = re.sub(r'\s*\n+\s*', ' ', text)

    # keep ellipses intact but add a space after if glued
    text = re.sub(r'(\.\.\.)(?=\S)', r'\1 ', text)

    # add a space after ., ?, ! when next visible char is a letter or an opening quote/paren
    # and NOT a decimal number (next char digit)
    text = re.sub(r'(?<!\.)'              # previous char is not a dot (avoid inside "...")
                  r'([.!?])'              # sentence end
                  r'(?=([""\'(\[]?[A-Za-z]))'  # next visible char is letter (maybe after quote/paren)
                  , r'\1 ', text)

    # optional: add space after : or ; if followed by a letter
    text = re.sub(r'([:;])(?=([""\'(\[]?[A-Za-z]))', r'\1 ', text)

    # de-glue common quote cases like: ."Word  .'Word  .)Word
    text = re.sub(r'([.!?][""\')\]])(?=\S)', r'\1 ', text)

    # collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

class TranscriptionProcessor:
    def __init__(self, model_size: str = "medium", segment_length: int = 7200):
        """
        Initialize the transcription processor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            segment_length: Length of audio segments in seconds (default: 2 hours for optimal speed/context balance)
        """
        self.model_manager = ModelManager()
        self.device = self.model_manager.get_device()
        self.model_size = model_size
        self.segment_length = segment_length
        
        print(f"Using device: {self.device}")
        
        # Apply CUDA optimizations if available
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            try:
                torch.cuda.set_per_process_memory_fraction(0.8)  # Slightly more conservative for stability
            except Exception as e:
                print(f"Warning: could not set memory fraction: {e}")
        
        # Load the specified model
        self.model = self.model_manager.get_model(model_size)
        
        # Apply FP16 optimization on GPU for better accuracy and speed
        if self.device == "cuda":
            self.model = self.model.half()
        
        print(f"Loaded Whisper '{model_size}' model")

    def get_model_recommendation(self, audio_path: str, target_quality: str = "balanced") -> dict:
        """
        Get intelligent model recommendation based on audio file duration and quality target.
        
        Args:
            audio_path: Path to the audio file
            target_quality: "fastest", "balanced", or "highest"
            
        Returns:
            Dictionary with recommendation details
        """
        try:
            # Analyze audio duration
            audio = AudioSegment.from_file(audio_path)
            duration_minutes = len(audio) / (1000 * 60)
            
            # Get recommendation from model manager
            recommendation = self.model_manager.recommend_model_size(
                duration_minutes=duration_minutes,
                target_quality=target_quality
            )
            
            recommendation['audio_duration_minutes'] = duration_minutes
            return recommendation
            
        except Exception as e:
            print(f"Error analyzing audio for model recommendation: {e}")
            # Return safe fallback
            return {
                "model_size": "medium",
                "estimated_time_minutes": 10.0,
                "memory_required_gb": 3.0,
                "quality_score": 0.9,
                "audio_duration_minutes": 0.0,
                "error": str(e)
            }

    def transcribe(self, audio_path: str, class_id: str, word_timestamps: bool = True) -> Optional[str]:
        """Transcribe audio using the notebook approach."""
        print(f"Processing audio to generate transcript JSON...")
        
        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000
            print(f"Total duration: {timedelta(seconds=int(total_duration))}")
            
            all_segments = []
            segment_times = list(range(0, int(total_duration), self.segment_length))
            print(f"Will process {len(segment_times)} segments: {segment_times}")
            
            for start_time in tqdm(segment_times, desc="Processing segments", unit="segment"):
                print(f"Starting segment at {start_time}s")
                duration = min(self.segment_length, total_duration - start_time)
                segment = audio[start_time*1000:(start_time+duration)*1000]
                temp_path = get_temp_path(f"temp_segment_{start_time}.wav")
                segment.export(temp_path, format="wav")
                
                try:
                    # Setup mixed precision context
                    if self.device == "cuda":
                        try:
                            ctx_mgr = torch.amp.autocast("cuda")
                        except Exception:
                            ctx_mgr = nullcontext()
                    else:
                        ctx_mgr = nullcontext()
                    
                    try:
                        with ctx_mgr:
                            print(f"Transcribing segment {start_time}s...")
                            result = self.model.transcribe(
                                temp_path,
                                word_timestamps=word_timestamps,
                                language='en',
                                task='transcribe',
                                fp16=(self.device == "cuda"),
                                condition_on_previous_text=True,
                                initial_prompt="This is a university lecture."
                            )
                        print(f"Completed segment {start_time}s, found {len(result['segments'])} segments")
                        
                        for seg in result["segments"]:
                            seg_start = float(seg["start"]) + start_time
                            seg_end = float(seg["end"]) + start_time
                            
                            # Handle words based on timestamp precision setting
                            words = []
                            if word_timestamps and seg.get("words"):
                                for w in seg.get("words", []):
                                    words.append({
                                        "word": w["word"].strip(),
                                        "start": float(w["start"]) + start_time,
                                        "end": float(w["end"]) + start_time
                                    })
                            
                            all_segments.append({
                                "start": seg_start,
                                "end": seg_end,
                                "text": normalize_sentence_spacing(seg["text"].strip()),
                                "words": words
                            })
                    
                    finally:
                        try:
                            Path(temp_path).unlink(missing_ok=True)
                        except:
                            pass
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()  # Ensure cleanup completes
                
                except Exception as e:
                    print(f"Error processing segment at {start_time}s: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not all_segments:
                print("Warning: No segments were transcribed.")
                return None
            
            transcript_path = get_temp_path(f"session_{class_id}_transcript.json")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump({"segments": sorted(all_segments, key=lambda x: x["start"])}, f, indent=2)
            
            print(f"Transcript JSON saved to: {transcript_path}")
            return transcript_path
            
        except Exception as e:
            print(f"Error in transcription process: {str(e)}")
            raise
    
    
