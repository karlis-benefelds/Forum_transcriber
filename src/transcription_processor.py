import whisper
import torch
import json
from datetime import timedelta
from pathlib import Path
from pydub import AudioSegment
from tqdm.auto import tqdm # Use tqdm.auto for better environment compatibility
from contextlib import nullcontext
from src.utils import get_temp_path

class TranscriptionProcessor:
    def __init__(self, segment_length=14400):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
        print("Loading Whisper model...")
        # Consider adding a model size parameter to init, e.g., "medium", "large"
        self.model = whisper.load_model("medium").to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.segment_length = segment_length

    def transcribe(self, audio_path, class_id):
        print(f"Processing audio to generate transcript JSON...")
        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000
            print(f"Total duration: {timedelta(seconds=int(total_duration))}")

            all_segments = []
            segment_times = range(0, int(total_duration), self.segment_length)

            for start_time in tqdm(segment_times, desc="Processing segments", unit="segment"):
                duration = min(self.segment_length, total_duration - start_time)
                segment = audio[start_time*1000:(start_time+duration)*1000]
                temp_path = get_temp_path(f"temp_segment_{start_time}.wav")
                segment.export(temp_path, format="wav")

                try:
                    # Updated deprecation: use torch.amp.autocast("cuda", ...)
                    use_amp = (self.device == "cuda")
                    ctx_mgr = torch.amp.autocast("cuda") if use_amp else nullcontext()
                except Exception:
                    # Fallback if torch.amp not available
                    class _Dummy:
                        def __enter__(self): pass
                        def __exit__(self, *a): pass
                    ctx_mgr = _Dummy()

                try:
                    with ctx_mgr:
                        result = self.model.transcribe(
                            temp_path,
                            word_timestamps=True,
                            language='en',
                            task='transcribe',
                            fp16=(self.device=="cuda"),
                            condition_on_previous_text=True,
                            initial_prompt="This is a university lecture."
                        )

                    for seg in result["segments"]:
                        seg_start = float(seg["start"]) + start_time
                        seg_end   = float(seg["end"]) + start_time
                        words = []
                        for w in seg.get("words", []):
                            words.append({
                                "word": w["word"].strip(),
                                "start": float(w["start"]) + start_time,
                                "end": float(w["end"]) + start_time
                            })
                        all_segments.append({
                            "start": seg_start,
                            "end": seg_end,
                            "text": seg["text"].strip(),
                            "words": words
                        })

                finally:
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except:
                        pass
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

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
