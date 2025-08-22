from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for, make_response
import os
import threading
import uuid
from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import with error handling
try:
    from src.main import process_lecture
except ImportError as e:
    print(f"Warning: Could not import process_lecture: {e}")
    process_lecture = None

try:
    from src.ai_chat import AIChat
except ImportError as e:
    print(f"Warning: Could not import AIChat: {e}")
    AIChat = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store processing status and chat sessions
processing_status = {}
chat_sessions = {}  # Store chat conversations by session ID

# Create directories for exports and session storage
os.makedirs('exports', exist_ok=True)
os.makedirs('sessions', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video_file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['video_file']
    curl_string = request.form.get('curl_string', '').strip()
    privacy_mode = request.form.get('privacy_mode', 'names')
    timestamp_precision = request.form.get('timestamp_precision', 'fast')
    # Optimal speed/quality balance
    model_size = 'medium'  # Sweet spot: 90% of large accuracy at 3x speed
    target_quality = 'highest'  # Keep highest quality processing
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not curl_string:
        return jsonify({'error': 'CURL string is required'}), 400
    
    # Validate file type
    if not file.filename.lower().endswith('.mp4'):
        return jsonify({'error': 'Please upload an MP4 file'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = f"{job_id}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Initialize processing status
    processing_status[job_id] = {
        'status': 'pending',
        'step': 'Preparing to process...',
        'filename': file.filename,
        'start_time': datetime.now(),
        'outputs': []
    }
    
    # Start processing in background
    thread = threading.Thread(
        target=process_video_background,
        args=(job_id, filepath, curl_string, privacy_mode, model_size, target_quality, timestamp_precision)
    )
    thread.start()
    
    return jsonify({'job_id': job_id, 'message': 'Upload successful, processing started'})

# Model recommendation endpoint removed - always use highest accuracy

def process_video_background(job_id, filepath, curl_string, privacy_mode, model_size, target_quality, timestamp_precision):
    try:
        processing_status[job_id]['status'] = 'processing'
        
        # Step 1/4: Fetch Forum data
        processing_status[job_id]['step'] = 'Step 1/4: Fetching Forum class events...'
        
        if process_lecture is None:
            raise Exception('Transcription functionality not available - missing dependencies')
        
        # Extract class_id from curl (this function exists in utils.py)
        from src.utils import extract_ids_from_curl
        ids = extract_ids_from_curl(curl_string)
        class_id = ids.get('class_id')
        
        if not class_id:
            raise Exception('Could not extract class ID from CURL string')
        
        # Process with user's chosen timestamp precision
        outputs = process_lecture_with_progress(job_id, filepath, class_id, curl_string, privacy_mode, None, 'medium', 'highest', timestamp_precision)
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['step'] = 'Processing complete!'
        processing_status[job_id]['outputs'] = outputs
        processing_status[job_id]['end_time'] = datetime.now()
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['step'] = f'Error: {str(e)}'
        processing_status[job_id]['error'] = str(e)

def process_lecture_with_progress(job_id, audio_path, class_id, curl_string, privacy_mode="names", user_terms=None, model_size=None, target_quality="balanced", timestamp_precision="fast"):
    """
    Modified version of process_lecture that updates processing status throughout.
    """
    from src.utils import clean_curl
    from src.audio_preprocessor import AudioPreprocessor
    from src.transcription_processor import TranscriptionProcessor
    from src.forum_data_fetcher import get_forum_events
    from src.report_generator import compile_transcript_to_pdf, compile_transcript_to_csv
    from src.performance_monitor import PerformanceMonitor
    import torch
    
    monitor = PerformanceMonitor()
    
    try:
        # Step 1/4: Forum data
        processing_status[job_id]['step'] = 'Step 1/4: Fetching Forum class events...'
        headers = clean_curl(curl_string)
        events_data = get_forum_events(class_id, headers, curl_string)
        
        # Step 2/4: Audio preprocessing
        processing_status[job_id]['step'] = 'Step 2/4: Preprocessing audio...'
        preprocessor = AudioPreprocessor()
        fixed_path = preprocessor.validate_and_fix_file(audio_path)
        
        # Step 3/4: Transcription
        processing_status[job_id]['step'] = 'Step 3/4: Transcribing audio...'
        
        # Optimal speed/quality balance
        model_size = 'medium'
        
        # Start performance monitoring
        monitor.start_monitoring({'model_size': model_size, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
        
        # Configure segment length and word timestamps based on user choice
        if timestamp_precision == "fast":
            segment_length = 3600  # 1 hour segments for speed
            use_word_timestamps = False
        else:  # detailed
            segment_length = 7200  # 2 hour segments for quality
            use_word_timestamps = True
        
        tp = TranscriptionProcessor(model_size=model_size, segment_length=segment_length)
        transcript_path = tp.transcribe(fixed_path, class_id, word_timestamps=use_word_timestamps)
        
        # Step 4/4: Generate outputs
        processing_status[job_id]['step'] = 'Step 4/4: Preparing outputs...'
        
        # Choose output modes
        modes = [privacy_mode] if privacy_mode in ("names", "ids") else ["names", "ids"]
        
        outputs = []
        for mode in modes:
            pdf_path = compile_transcript_to_pdf(class_id, headers, privacy_mode=mode)
            csv_path = compile_transcript_to_csv(class_id, headers, privacy_mode=mode)
            outputs.append((mode, str(pdf_path), str(csv_path)))
        
        # Clean up temp WAV (but keep original downloaded file if different)
        try:
            if fixed_path and Path(fixed_path).exists() and (str(fixed_path) != str(audio_path)):
                Path(fixed_path).unlink(missing_ok=True)
        except Exception:
            pass  # Don't fail the whole process for cleanup issues
        
        monitor.stop_monitoring()
        return outputs
        
    except Exception as e:
        monitor.stop_monitoring()
        raise e

@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = processing_status[job_id].copy()
    
    # Calculate elapsed time
    if 'start_time' in status:
        elapsed = datetime.now() - status['start_time']
        status['elapsed_time'] = str(elapsed).split('.')[0]  # Remove microseconds
    
    return jsonify(status)

@app.route('/results/<job_id>')
def results_page(job_id):
    if job_id not in processing_status:
        return "Job not found", 404
    
    return render_template('results.html', job_id=job_id)

@app.route('/download/<job_id>/<file_type>/<privacy_mode>')
def download_file(job_id, file_type, privacy_mode):
    if job_id not in processing_status:
        return "Job not found", 404
    
    status = processing_status[job_id]
    if status['status'] != 'completed':
        return "Processing not complete", 400
    
    # Find the correct output file
    for mode, pdf_path, csv_path in status['outputs']:
        if mode == privacy_mode:
            if file_type == 'pdf':
                return send_file(pdf_path, as_attachment=True)
            elif file_type == 'csv':
                return send_file(csv_path, as_attachment=True)
    
    return "File not found", 404

# AI Chat Routes
@app.route('/chat/upload', methods=['POST'])
def upload_transcripts_for_chat():
    """Upload transcript files for AI analysis"""
    if not AIChat:
        return jsonify({'error': 'AI Chat service not available'}), 500
    
    if 'transcript_files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('transcript_files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        # Initialize AI Chat
        ai_chat = AIChat()
        
        # Process uploaded files
        transcript_content = ai_chat.process_uploaded_files(files)
        
        # Create chat session
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = {
            'transcript_content': transcript_content,
            'conversation_history': [],
            'filenames': [f.filename for f in files],
            'created_at': datetime.now()
        }
        
        # Get context information
        context_info = ai_chat.get_context_info(transcript_content)
        
        return jsonify({
            'session_id': session_id,
            'context_info': context_info,
            'filenames': chat_sessions[session_id]['filenames']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/analyze/<session_id>', methods=['POST'])
def initial_analysis(session_id):
    """Generate initial structured analysis"""
    if not AIChat or session_id not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        ai_chat = AIChat()
        session = chat_sessions[session_id]
        
        # Generate initial analysis
        analysis = ai_chat.generate_initial_analysis(session['transcript_content'])
        
        # Add to conversation history
        session['conversation_history'].extend([
            {'role': 'user', 'content': 'Please provide an initial comprehensive analysis of this transcript.'},
            {'role': 'assistant', 'content': analysis}
        ])
        
        return jsonify({'analysis': analysis})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/message/<session_id>', methods=['POST'])
def chat_message(session_id):
    """Send message in chat conversation"""
    if not AIChat or session_id not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    try:
        ai_chat = AIChat()
        session = chat_sessions[session_id]
        
        # Get AI response
        ai_response = ai_chat.chat_with_transcript(
            session['transcript_content'],
            session['conversation_history'],
            user_message
        )
        
        # Update conversation history
        session['conversation_history'].extend([
            {'role': 'user', 'content': user_message},
            {'role': 'assistant', 'content': ai_response}
        ])
        
        return jsonify({'response': ai_response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/export/<session_id>')
def export_conversation_markdown(session_id):
    """Export chat conversation as markdown file"""
    if not AIChat or session_id not in chat_sessions:
        return "Session not found", 404
    
    try:
        ai_chat = AIChat()
        session = chat_sessions[session_id]
        
        # Generate export content
        transcript_filename = " & ".join(session['filenames'])
        export_content = ai_chat.export_conversation_markdown(
            session['conversation_history'],
            transcript_filename
        )
        
        # Create response
        response = make_response(export_content)
        response.headers['Content-Type'] = 'text/markdown'
        response.headers['Content-Disposition'] = f'attachment; filename="ai_analysis_{session_id[:8]}.md"'
        
        return response
        
    except Exception as e:
        return f"Error exporting conversation: {str(e)}", 500

@app.route('/chat/export/<session_id>/pdf')
def export_conversation_pdf(session_id):
    """Export chat conversation as PDF file"""
    if not AIChat or session_id not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        ai_chat = AIChat()
        session = chat_sessions[session_id]
        
        # Check if conversation history exists and is not empty
        if not session.get('conversation_history') or len(session['conversation_history']) == 0:
            return jsonify({'error': 'No conversation to export'}), 400
        
        # Generate PDF
        transcript_filename = " & ".join(session['filenames'])
        pdf_filename = f"ai_analysis_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join('exports', pdf_filename)
        
        # Ensure exports directory exists
        os.makedirs('exports', exist_ok=True)
        
        ai_chat.export_conversation_pdf(
            session['conversation_history'],
            transcript_filename,
            pdf_path
        )
        
        if not os.path.exists(pdf_path):
            return jsonify({'error': 'PDF generation failed'}), 500
        
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
        
    except Exception as e:
        print(f"PDF Export Error: {str(e)}")  # Add logging
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return jsonify({'error': f'Error exporting PDF: {str(e)}'}), 500

@app.route('/chat/sessions')
def list_chat_sessions():
    """List all available chat sessions"""
    sessions_list = []
    for session_id, session_data in chat_sessions.items():
        sessions_list.append({
            'session_id': session_id,
            'filenames': session_data['filenames'],
            'created_at': session_data['created_at'].isoformat(),
            'message_count': len(session_data['conversation_history']),
            'job_id': session_data.get('job_id')
        })
    
    # Sort by creation date (newest first)
    sessions_list.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify({'sessions': sessions_list})

@app.route('/chat/session/<session_id>')
def get_chat_session(session_id):
    """Get a specific chat session data"""
    if session_id not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session = chat_sessions[session_id]
    return jsonify({
        'session_id': session_id,
        'filenames': session['filenames'],
        'created_at': session['created_at'].isoformat(),
        'conversation_history': session['conversation_history'],
        'job_id': session.get('job_id')
    })

@app.route('/chat/resume/<session_id>', methods=['POST'])
def resume_chat_session(session_id):
    """Resume a previous chat session"""
    if not AIChat or session_id not in chat_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    try:
        ai_chat = AIChat()
        session = chat_sessions[session_id]
        
        # Get context information
        context_info = ai_chat.get_context_info(session['transcript_content'])
        
        return jsonify({
            'session_id': session_id,
            'context_info': context_info,
            'filenames': session['filenames'],
            'conversation_history': session['conversation_history']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/save_and_analyze/<job_id>', methods=['POST'])
def save_and_analyze(job_id):
    """Save transcript from results and start AI analysis"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = processing_status[job_id]
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not complete'}), 400
    
    try:
        if not AIChat:
            return jsonify({'error': 'AI Chat service not available'}), 500
        
        ai_chat = AIChat()
        
        # Read the generated transcript content (use first available output)
        pdf_path = status['outputs'][0][1]  # First output's PDF path
        
        with open(pdf_path, 'rb') as f:
            transcript_content = ai_chat.extract_text_from_pdf(f)
        
        # Create chat session
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = {
            'transcript_content': transcript_content,
            'conversation_history': [],
            'filenames': [status['filename']],
            'created_at': datetime.now(),
            'job_id': job_id
        }
        
        # Get context information
        context_info = ai_chat.get_context_info(transcript_content)
        
        return jsonify({
            'session_id': session_id,
            'context_info': context_info,
            'filename': status['filename']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8001)