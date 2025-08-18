from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
import os
import threading
import uuid
from pathlib import Path
import json
from datetime import datetime

# Import with error handling
try:
    from src.main import process_lecture
except ImportError as e:
    print(f"Warning: Could not import process_lecture: {e}")
    process_lecture = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store processing status
processing_status = {}

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
        args=(job_id, filepath, curl_string, privacy_mode)
    )
    thread.start()
    
    return jsonify({'job_id': job_id, 'message': 'Upload successful, processing started'})

def process_video_background(job_id, filepath, curl_string, privacy_mode):
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['step'] = 'Processing video...'
        
        if process_lecture is None:
            raise Exception('Transcription functionality not available - missing dependencies')
        
        # Extract class_id from curl (this function exists in utils.py)
        from src.utils import extract_ids_from_curl
        ids = extract_ids_from_curl(curl_string)
        class_id = ids.get('class_id')
        
        if not class_id:
            raise Exception('Could not extract class ID from CURL string')
        
        # Process the lecture
        outputs = process_lecture(filepath, class_id, curl_string, privacy_mode)
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['step'] = 'Processing complete!'
        processing_status[job_id]['outputs'] = outputs
        processing_status[job_id]['end_time'] = datetime.now()
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['step'] = f'Error: {str(e)}'
        processing_status[job_id]['error'] = str(e)

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)