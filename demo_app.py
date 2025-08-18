from flask import Flask, request, render_template, jsonify, send_file
import os
import threading
import uuid
from pathlib import Path
import json
from datetime import datetime
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('outputs', exist_ok=True)

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
        target=demo_process_video,
        args=(job_id, filepath, curl_string, privacy_mode)
    )
    thread.start()
    
    return jsonify({'job_id': job_id, 'message': 'Upload successful, processing started'})

def demo_process_video(job_id, filepath, curl_string, privacy_mode):
    """Demo version that simulates the transcription process"""
    try:
        # Simulate processing steps
        steps = [
            ('Fetching Forum class events...', 3),
            ('Preprocessing audio...', 5),
            ('Transcribing audio...', 15),
            ('Generating reports...', 4),
            ('Finalizing outputs...', 2)
        ]
        
        for step_name, duration in steps:
            processing_status[job_id]['status'] = 'processing'
            processing_status[job_id]['step'] = step_name
            time.sleep(duration)
        
        # Create demo output files
        create_demo_outputs(job_id, privacy_mode)
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['step'] = 'Processing complete!'
        processing_status[job_id]['end_time'] = datetime.now()
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['step'] = f'Error: {str(e)}'
        processing_status[job_id]['error'] = str(e)

def create_demo_outputs(job_id, privacy_mode):
    """Create demo PDF and CSV files"""
    modes = [privacy_mode] if privacy_mode in ['names', 'ids'] else ['names', 'ids']
    outputs = []
    
    for mode in modes:
        # Create demo PDF content
        pdf_content = f"""Demo Transcript - {mode.title()} Mode
        
This is a demonstration transcript file.
In the real application, this would contain:

- Automated speech transcription from the uploaded video
- Student participation data from Forum
- Timestamps and speaker identification
- Privacy mode: {mode}

Generated at: {datetime.now()}
Job ID: {job_id}
"""
        
        # Create demo CSV content
        csv_content = f"""timestamp,speaker,content,privacy_mode
00:01:23,{('Student_123' if mode == 'ids' else 'John Doe')},This is a sample transcript line,{mode}
00:02:45,Professor,Welcome to today's lecture,{mode}
00:03:12,{('Student_456' if mode == 'ids' else 'Jane Smith')},I have a question about the assignment,{mode}
"""
        
        # Save demo files
        pdf_path = f"outputs/{job_id}_{mode}_transcript.pdf"
        csv_path = f"outputs/{job_id}_{mode}_transcript.csv"
        
        # Write demo content (as text files for demo purposes)
        with open(pdf_path.replace('.pdf', '.txt'), 'w') as f:
            f.write(pdf_content)
        
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        outputs.append((mode, pdf_path.replace('.pdf', '.txt'), csv_path))
    
    processing_status[job_id]['outputs'] = outputs

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
                return send_file(pdf_path, as_attachment=True, download_name=f"transcript_{privacy_mode}.txt")
            elif file_type == 'csv':
                return send_file(csv_path, as_attachment=True, download_name=f"transcript_{privacy_mode}.csv")
    
    return "File not found", 404

if __name__ == '__main__':
    print("\n🎓 Class Transcriber Demo Server")
    print("=" * 40)
    print("📹 Upload MP4 videos for transcription")
    print("🔗 Provide Forum cURL for student data") 
    print("📄 Download transcripts in PDF/CSV formats")
    print("=" * 40)
    print("🌐 Open: http://localhost:8001")
    print("⚠️  This is a DEMO version with simulated processing")
    print("=" * 40)
    
    app.run(debug=True, host='0.0.0.0', port=8001)