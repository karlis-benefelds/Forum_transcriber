import argparse
from pathlib import Path
import torch
import gc
import sys
import os

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.utils import clean_curl, extract_ids_from_curl, get_temp_path
from src.audio_preprocessor import AudioPreprocessor
from src.transcription_processor import TranscriptionProcessor
from src.forum_data_fetcher import get_forum_events
from src.report_generator import compile_transcript_to_pdf, compile_transcript_to_csv
from src.performance_monitor import PerformanceMonitor
# Note: create_simplified_csv and create_simplified_transcript are not directly used in the main process_lecture pipeline
# but are kept in report_generator.py if needed for fallback scenarios not handled by this main function.

def process_lecture(audio_path, class_id, curl_string, privacy_mode="names", user_terms=None, performance_mode="balanced"):
    """
    Main pipeline to process lecture audio, fetch forum data, transcribe, and generate reports.
    Args:
        audio_path (str): Path or URL to the audio file.
        class_id (str): The class ID.
        curl_string (str): The cURL string for Forum API authentication.
        privacy_mode (str): 'names', 'ids', or 'both' for report generation.
        user_terms (list): List of custom terms to preserve spellings (currently unused).
        performance_mode (str): 'fast', 'balanced', or 'accurate' processing mode.
    Returns:
        list: A list of tuples, each containing (privacy_mode, pdf_path, csv_path).
    """
    monitor = PerformanceMonitor()
    
    try:
        print("Step 1/4: Fetching Forum class events...")
        headers = clean_curl(curl_string)
        events_data = get_forum_events(class_id, headers, curl_string)

        print("\nStep 2/4: Preprocessing audio...")
        preprocessor = AudioPreprocessor()
        fixed_path = preprocessor.validate_and_fix_file(audio_path)
        
        # Get audio duration for performance monitoring
        from pydub import AudioSegment
        audio = AudioSegment.from_file(fixed_path)
        audio_duration = len(audio) / 1000
        
        # Determine optimal settings based on performance mode and duration
        model_size, segment_length, max_workers = _get_optimal_settings(
            audio_duration, performance_mode
        )

        print(f"\nStep 3/4: Transcribing (Mode: {performance_mode})...")
        print(f"Settings: model={model_size}, segment_length={segment_length}s, workers={max_workers}")
        
        # Start performance monitoring
        monitor.start_monitoring({
            'audio_duration': audio_duration,
            'model_size': model_size,
            'device': 'auto',  # Will be determined by TranscriptionProcessor
            'total_segments': max(1, int(audio_duration // segment_length))
        })
        
        tp = TranscriptionProcessor(
            model_size=model_size,
            segment_length=segment_length,
            max_workers=max_workers
        )
        
        transcript_path = tp.transcribe(fixed_path, class_id, parallel=True)

        print("\nStep 4/4: Preparing outputs...")

        # Choose output modes
        modes = [privacy_mode] if privacy_mode in ("names", "ids") else ["names", "ids"]

        outputs = []
        for mode in modes:
            pdf_path = compile_transcript_to_pdf(class_id, headers, privacy_mode=mode)
            csv_path = compile_transcript_to_csv(class_id, headers, privacy_mode=mode)
            outputs.append((mode, pdf_path, csv_path))

        # Accuracy caution
        print("\n⚠️  Accuracy caution: Do not rely solely on this transcript. Manually verify key information.")

        # Clean up temp WAV (but keep original downloaded file if different)
        try:
            if fixed_path and Path(fixed_path).exists() and (str(fixed_path) != str(audio_path)):
                # This attempts to delete the converted WAV file, not the original download
                Path(fixed_path).unlink()
                print(f"Cleaned up temporary file: {fixed_path}")

            # Also delete the temporary downloaded file if it was a URL
            downloaded_file_name = get_temp_path("input_downloaded.mp4")
            if Path(downloaded_file_name).exists() and downloaded_file_name != audio_path:
                Path(downloaded_file_name).unlink()
                print(f"Cleaned up temporary downloaded file: {downloaded_file_name}")

            # Clean up temporary JSON files
            temp_json_transcript = Path(get_temp_path(f"session_{class_id}_transcript.json"))
            if temp_json_transcript.exists():
                temp_json_transcript.unlink()
                print(f"Cleaned up temporary JSON: {temp_json_transcript}")

            temp_json_events = Path(get_temp_path(f"session_{class_id}_events.json"))
            if temp_json_events.exists():
                temp_json_events.unlink()
                print(f"Cleaned up temporary JSON: {temp_json_events}")

        except Exception as cleanup_error:
            print(f"Note: Could not clean up temporary files: {str(cleanup_error)}")

        # Stop monitoring and get performance summary
        performance_summary = monitor.stop_monitoring()
        if performance_summary:
            print(f"\n📊 Performance Summary:")
            print(f"  Processing time: {performance_summary['total_processing_time']:.1f}s")
            if performance_summary.get('real_time_factor'):
                print(f"  Speed: {performance_summary['real_time_factor']:.1f}x real-time")
            print(f"  Peak CPU: {performance_summary['peak_cpu_percent']:.1f}%")
            print(f"  Peak Memory: {performance_summary['peak_memory_percent']:.1f}%")
            if performance_summary['peak_gpu_memory_gb'] > 0:
                print(f"  Peak GPU Memory: {performance_summary['peak_gpu_memory_gb']:.1f}GB")

        return outputs

    except Exception as e:
        monitor.stop_monitoring()
        print(f"\nERROR: {str(e)}")
        if "MP4" in str(e) and audio_path.lower().endswith('.mp4'):
            print("\nThere was a problem with your MP4 file. Suggestions:")
            print("1. Try converting it to MP3 or WAV locally before uploading")
            print("2. Use a screen recorder to re-record the audio")
            print("3. Contact Forum support about MP4 download issues")
        else:
            print("\nTranscription failed. Please try again with a different file or path.")
        return []

def _get_optimal_settings(audio_duration_seconds: float, performance_mode: str = "balanced"):
    """
    Determine optimal transcription settings based on audio duration and performance mode.
    
    Args:
        audio_duration_seconds: Duration of audio in seconds
        performance_mode: 'fast', 'balanced', or 'accurate'
    
    Returns:
        tuple: (model_size, segment_length, max_workers)
    """
    duration_minutes = audio_duration_seconds / 60
    
    # Import to check available devices
    try:
        import torch
        has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    except ImportError:
        has_gpu = False
    
    if performance_mode == "fast":
        if duration_minutes > 90:
            model_size = "base"
            segment_length = 1800  # 30 minutes
            max_workers = 4 if has_gpu else 2
        elif duration_minutes > 30:
            model_size = "small"
            segment_length = 3600  # 1 hour
            max_workers = 3 if has_gpu else 2
        else:
            model_size = "small"
            segment_length = 7200  # 2 hours
            max_workers = 2
    
    elif performance_mode == "accurate":
        if has_gpu:
            model_size = "large" if duration_minutes < 120 else "medium"
            segment_length = 3600  # 1 hour
            max_workers = 2
        else:
            model_size = "medium"
            segment_length = 7200  # 2 hours  
            max_workers = 1
    
    else:  # balanced
        if duration_minutes > 120:
            model_size = "small" if not has_gpu else "medium"
            segment_length = 1800  # 30 minutes
            max_workers = 4 if has_gpu else 2
        elif duration_minutes > 60:
            model_size = "medium" if has_gpu else "small"
            segment_length = 3600  # 1 hour
            max_workers = 3 if has_gpu else 2
        else:
            model_size = "medium"
            segment_length = 7200  # 2 hours
            max_workers = 2
    
    return model_size, segment_length, max_workers

if __name__ == "__main__":
    # Command-line argument parsing instead of input() for a cleaner app structure
    parser = argparse.ArgumentParser(description="Transcribe lecture audio and generate reports.")
    parser.add_argument('--curl', type=str, required=True, help="Forum cURL string for API authentication.")
    parser.add_argument('--audio_path', type=str, required=True, help="Path or URL to the media file.")
    parser.add_argument('--privacy_mode', type=str, choices=['names', 'ids', 'both'], default='names',
                        help="Student name privacy mode: 'names', 'ids', or 'both'.")
    parser.add_argument('--class_id', type=str, help="Optional: Class ID. Will try to auto-derive from cURL if not provided.")
    parser.add_argument('--user_terms', type=str, default="", help="Optional: Comma-separated custom terms to preserve spellings.")

    args = parser.parse_args()

    raw_curl = args.curl
    AUDIO_PATH = args.audio_path
    PRIVACY_MODE = args.privacy_mode
    USER_TERMS = [t.strip() for t in args.user_terms.split(",") if t.strip()]

    # Auto-derive Class ID from the cURL (uses existing helper)
    _ids = extract_ids_from_curl(raw_curl)
    CLASS_ID = args.class_id or _ids.get("class_id")

    if not CLASS_ID:
        print("Error: Could not determine Class ID from cURL or command-line argument. Please provide --class_id.")
        exit(1)

    # Summary
    print("Thanks! Summary:\n")
    print(f"Class ID (detected): {CLASS_ID}")
    print(f"Media: {AUDIO_PATH}")
    print(f"Privacy mode: {PRIVACY_MODE}")
    if USER_TERMS:
        print(f"Custom terms: {USER_TERMS}")
    print("\nStarting the transcript generation process…")
    print("⏳ This can take a while depending on file length and model size...")

    # Run the main processing function
    outs = process_lecture(AUDIO_PATH, CLASS_ID, raw_curl, PRIVACY_MODE, USER_TERMS)

    # CUDA cleanup
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    gc.collect()
    print("CUDA cache cleared")

    # Pretty print results
    if outs:
        if len(outs) == 1:
            mode, pdfp, csvp = outs[0]
            print(f"\nSuccess! Your transcripts are ready ({mode}):")
            print(f"PDF: {pdfp}")
            print(f"CSV: {csvp}")
        else:
            # both
            print("\nSuccess! Your transcripts are ready (both privacy modes):")
            for mode, pdfp, csvp in outs:
                print(f"PDF ({mode}): {pdfp}")
                print(f"CSV ({mode}): {csvp}")
    else:
        print("\nTranscription process did not complete successfully or no outputs were generated.")
