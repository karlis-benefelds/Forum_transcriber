import whisper
import torch
import json
import concurrent.futures
import time
from datetime import timedelta
from pathlib import Path
from pydub import AudioSegment
from tqdm.auto import tqdm
from contextlib import nullcontext
from typing import List, Dict, Tuple, Optional
from src.utils import get_temp_path
from src.model_manager import ModelManager

class TranscriptionProcessor:
    def __init__(self, model_size: str = "medium", segment_length: int = 7200, max_workers: int = 2):
        """
        Initialize the transcription processor with optimizations.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            segment_length: Length of audio segments in seconds (default: 2 hours)
            max_workers: Maximum parallel workers for processing (default: 2)
        """
        self.model_manager = ModelManager()
        self.model_size = model_size
        self.segment_length = segment_length
        self.max_workers = max_workers
        self.device = self.model_manager.get_device()
        
        print(f"TranscriptionProcessor initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model size: {model_size}")
        print(f"  Segment length: {segment_length}s")
        print(f"  Max workers: {max_workers}")
        
        # Pre-load the model
        self.model_manager.get_model(model_size)

    def transcribe(self, audio_path: str, class_id: str, parallel: bool = True) -> Optional[str]:
        """Transcribe audio with optimized parallel processing."""
        print(f"Processing audio to generate transcript JSON...")
        start_time = time.time()
        
        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000
            print(f"Total duration: {timedelta(seconds=int(total_duration))}")
            
            # Auto-recommend model size based on duration
            recommended_model = self.model_manager.recommend_model_size(
                total_duration / 60, "balanced"
            )
            if recommended_model != self.model_size:
                print(f"Recommended model size: {recommended_model} (current: {self.model_size})")
            
            # Prepare segments
            segment_info = self._prepare_segments(audio, total_duration, class_id)
            
            if parallel and len(segment_info) > 1 and self.max_workers > 1:
                all_segments = self._transcribe_parallel(segment_info)
            else:
                all_segments = self._transcribe_sequential(segment_info)
            
            if not all_segments:
                print("Warning: No segments were transcribed.")
                return None
            
            # Save results
            transcript_path = get_temp_path(f"session_{class_id}_transcript.json")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "segments": sorted(all_segments, key=lambda x: x["start"]),
                    "processing_time": time.time() - start_time,
                    "model_size": self.model_size,
                    "device": self.device
                }, f, indent=2)
            
            processing_time = time.time() - start_time
            print(f"Transcript JSON saved to: {transcript_path}")
            print(f"Total processing time: {processing_time:.2f}s ({processing_time/total_duration:.2f}x real-time)")
            
            return transcript_path
            
        except Exception as e:
            print(f"Error in transcription process: {str(e)}")
            raise
    
    def _prepare_segments(self, audio: AudioSegment, total_duration: float, class_id: str) -> List[Dict]:
        """Prepare audio segments for processing."""
        segment_info = []
        segment_times = range(0, int(total_duration), self.segment_length)
        
        for i, start_time in enumerate(segment_times):
            duration = min(self.segment_length, total_duration - start_time)
            segment = audio[start_time*1000:(start_time+duration)*1000]
            temp_path = get_temp_path(f"temp_segment_{class_id}_{i}_{start_time}.wav")
            segment.export(temp_path, format="wav")
            
            segment_info.append({
                'index': i,
                'start_time': start_time,
                'duration': duration,
                'temp_path': temp_path
            })
        
        print(f"Prepared {len(segment_info)} segments for processing")
        return segment_info
    
    def _transcribe_parallel(self, segment_info: List[Dict]) -> List[Dict]:
        """Transcribe segments in parallel."""
        print(f"Processing {len(segment_info)} segments with {self.max_workers} workers...")
        
        all_segments = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_segment = {
                executor.submit(self._transcribe_segment, seg_info): seg_info 
                for seg_info in segment_info
            }
            
            # Collect results with progress bar
            with tqdm(total=len(segment_info), desc="Transcribing segments", unit="segment") as pbar:
                for future in concurrent.futures.as_completed(future_to_segment):
                    seg_info = future_to_segment[future]
                    try:
                        segments = future.result()
                        all_segments.extend(segments)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Segment {seg_info['index']} failed: {str(e)}")
                    finally:
                        # Cleanup temp file
                        try:
                            Path(seg_info['temp_path']).unlink(missing_ok=True)
                        except:
                            pass
        
        return all_segments
    
    def _transcribe_sequential(self, segment_info: List[Dict]) -> List[Dict]:
        """Transcribe segments sequentially (fallback)."""
        print(f"Processing {len(segment_info)} segments sequentially...")
        
        all_segments = []
        for seg_info in tqdm(segment_info, desc="Processing segments", unit="segment"):
            try:
                segments = self._transcribe_segment(seg_info)
                all_segments.extend(segments)
            except Exception as e:
                print(f"Segment {seg_info['index']} failed: {str(e)}")
            finally:
                # Cleanup temp file
                try:
                    Path(seg_info['temp_path']).unlink(missing_ok=True)
                except:
                    pass
        
        return all_segments
    
    def _transcribe_segment(self, seg_info: Dict) -> List[Dict]:
        """Transcribe a single segment."""
        model = self.model_manager.get_model(self.model_size)
        temp_path = seg_info['temp_path']
        start_time = seg_info['start_time']
        
        try:
            # Setup mixed precision context
            if self.device in ["cuda", "mps"]:
                try:
                    ctx_mgr = torch.amp.autocast(self.device)
                except Exception:
                    ctx_mgr = nullcontext()
            else:
                ctx_mgr = nullcontext()
            
            with ctx_mgr:
                result = model.transcribe(
                    temp_path,
                    word_timestamps=True,
                    language='en',
                    task='transcribe',
                    fp16=(self.device in ["cuda", "mps"]),
                    condition_on_previous_text=True,
                    initial_prompt="This is a university lecture."
                )
            
            # Process segments
            segments = []
            for seg in result["segments"]:
                seg_start = float(seg["start"]) + start_time
                seg_end = float(seg["end"]) + start_time
                words = []
                
                for w in seg.get("words", []):
                    words.append({
                        "word": w["word"].strip(),
                        "start": float(w["start"]) + start_time,
                        "end": float(w["end"]) + start_time
                    })
                
                segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg["text"].strip(),
                    "words": words
                })
            
            return segments
            
        finally:
            # Cleanup GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        stats = {
            'device': self.device,
            'model_size': self.model_size,
            'segment_length': self.segment_length,
            'max_workers': self.max_workers
        }
        
        # Add memory stats
        memory_stats = self.model_manager.get_memory_usage()
        stats.update(memory_stats)
        
        return stats
    
    def update_settings(self, **kwargs):
        """Update processor settings dynamically."""
        if 'model_size' in kwargs:
            self.model_size = kwargs['model_size']
        if 'segment_length' in kwargs:
            self.segment_length = kwargs['segment_length']
        if 'max_workers' in kwargs:
            self.max_workers = kwargs['max_workers']
        
        print(f"Updated settings: model={self.model_size}, segment_length={self.segment_length}, workers={self.max_workers}")
