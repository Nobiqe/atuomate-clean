# Import necessary libraries for file handling, Flask, threading, scheduling, and utilities
import os
import random
import shutil
import tracebackraries
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from id_card_processor import process_id_card
from functools import wraps
import magic  # For validating file MIME types
# Initialize Flask app with static folder for serving output images
app = Flask(__name__, static_folder='output_images', static_url_path='/output_images')
# Configuration settings for the app
app.config['UPLOAD_FOLDER'] = 'uploads'# Directory for uploaded files
app.config['OUTPUT_FOLDER'] = 'output_images'# Directory for processed outputs
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}# Allowed image formats
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'mov', 'webm'}# Allowed video formats
app.config['MAX_IMAGE_SIZE'] = 16 * 1024 * 1024  # 16MB # Max image size: 16MB
app.config['MAX_VIDEO_SIZE'] = 30 * 1024 * 1024  # 30MB # Max video size: 30MB
app.config['JOB_RETENTION_HOURS'] = 24 # Time to retain job data (24 hours)
app.config['API_KEY'] = 'your-secret-api-key' # API key for authentication
MAX_WORKERS = 4 # Max number of concurrent processing threads
# Create upload and output directories if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
# Initialize thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Global dictionary to store job statuses
processing_jobs = {}

# Function to check if a file has an allowed extension
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Decorator to enforce API key authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check API key in form data or headers
        api_key = request.form.get('api_key') or request.headers.get('X-API-Key')
        if api_key != app.config['API_KEY']:
            return jsonify({'status': 'error', 'message': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function
# Function to validate file content using MIME types
def validate_file_content(file, allowed_mime_types):
    mime = magic.Magic(mime=True)
    file.seek(0)
    mime_type = mime.from_buffer(file.read(1024))  # Read first 1024 bytes
    file.seek(0)  # Reset file pointer
    return mime_type in allowed_mime_types

# Function to process uploaded images and video for a job
def process_job(job_id, first_image_bytes, second_image_bytes, video_bytes=None):
    try:
        # Create output directory for this job
        job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        os.makedirs(job_output_dir, exist_ok=True)
        log_file = os.path.join(job_output_dir, "text_log.log")  # Log file for processing

        # Update job status to 'processing'
        processing_jobs[job_id].update({'status': 'processing', 'message': 'Processing images...'})

        # Process ID card images using external function
        result = process_id_card(first_image_bytes, second_image_bytes, job_output_dir, log_file)

        # Save verification video if provided
        if video_bytes:
            video_path = os.path.join(job_output_dir, "verification_video.mp4")
            with open(video_path, 'wb') as f:
                f.write(video_bytes)  # Write video to file
            result["image_paths"]["verification_video"] = f"/outputs/{job_id}/verification_video.mp4"

        # Placeholder for liveness and identity verification (to be replaced with real logic)
        result["data"]["is_alive"] = True  # Dummy value
        result["data"]["is_same_person"] = True  # Dummy value

        # Update file paths to web-accessible URLs
        for key, path in result["image_paths"].items():
            if path:
                result["image_paths"][key] = f"/outputs/{job_id}/{os.path.basename(path)}"

        # Update job status with final results
        processing_jobs[job_id].update({
            'status': result['status'],
            'message': result['message'],
            'data': result['data'],
            'image_paths': result['image_paths'],
            'completed_at': str(datetime.now())
        })
    except Exception as e:
        # Handle errors and update job status
        processing_jobs[job_id].update({
            'status': 'error',
            'message': f"Error: {str(e)}",
            'error_details': traceback.format_exc()
        })
        print(f"Job {job_id} failed: {traceback.format_exc()}")

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')  # Render index.html

# Route to display results for a specific job
@app.route('/results/<job_id>')
def results(job_id):
    if job_id not in processing_jobs:
        return render_template('error.html', message="Job not found or has expired"), 404
    job = processing_jobs[job_id]  # Get job details
    return render_template('results.html', job=job, job_id=job_id)  # Render results page

# Handle 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', code="404", message="Page not found"), 404

# Handle 500 errors
@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code="500", message="Server error occurred"), 500

# Route to handle file uploads
@app.route('/api/upload', methods=['POST'])
@require_api_key
def api_upload():
    # Check for required files
    required_files = ['first_image', 'second_image', 'verification_video']
    if not all(f in request.files for f in required_files):
        return render_template('error.html', message="All files (front image, back image, video) are required"), 400

    # Get uploaded files
    first_file = request.files['first_image']
    second_file = request.files['second_image']
    video_file = request.files['verification_video']

    # Validate file extensions
    if not allowed_file(first_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']) or \
       not allowed_file(second_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
        return render_template('error.html', message="Invalid image file type (only PNG, JPG, JPEG allowed)"), 400

    if not allowed_file(video_file.filename, app.config['ALLOWED_VIDEO_EXTENSIONS']):
        return render_template('error.html', message="Invalid video file type (only MP4, MOV, WebM allowed)"), 400

    # Validate file sizes
    first_file.seek(0, os.SEEK_END)
    second_file.seek(0, os.SEEK_END)
    video_file.seek(0, os.SEEK_END)
    if first_file.tell() > app.config['MAX_IMAGE_SIZE'] or second_file.tell() > app.config['MAX_IMAGE_SIZE']:
        return render_template('error.html', message="Image file size exceeds 16MB"), 400
    if video_file.tell() > app.config['MAX_VIDEO_SIZE']:
        return render_template('error.html', message="Video file size exceeds 30MB"), 400

    # Reset file pointers
    first_file.seek(0)
    second_file.seek(0)
    video_file.seek(0)

    # Validate file content types
    image_mime_types = {'image/jpeg', 'image/png'}
    video_mime_types = {'video/mp4', 'video/quicktime', 'video/webm'}
    if not validate_file_content(first_file, image_mime_types) or \
       not validate_file_content(second_file, image_mime_types):
        return render_template('error.html', message="Invalid image content"), 400
    if not validate_file_content(video_file, video_mime_types):
        return render_template('error.html', message="Invalid video content"), 400

    # Generate unique job ID based on timestamp and random number
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(1000, 9999)}"

    # Read file contents
    first_bytes = first_file.read()
    second_bytes = second_file.read()
    video_bytes = video_file.read()

    # Initialize job status in processing_jobs
    processing_jobs[job_id] = {
        'status': 'pending',
        'message': 'Processing started',
        'created_at': str(datetime.now()),
        'first_image': secure_filename(first_file.filename),
        'second_image': secure_filename(second_file.filename),
        'verification_video': secure_filename(video_file.filename)
    }

    # Submit job for asynchronous processing
    executor.submit(process_job, job_id, first_bytes, second_bytes, video_bytes)

    # Redirect to results page
    return redirect(url_for('results', job_id=job_id))

# Route to check job status via API
@app.route('/api/job/<job_id>', methods=['GET'])
@require_api_key
def api_job_status(job_id):
    if job_id not in processing_jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404
    # Return job details, excluding sensitive error details
    return jsonify({k: v for k, v in processing_jobs[job_id].items() if k != 'error_details'})

# Route to serve output files (images, videos)
@app.route('/outputs/<job_id>/<filename>')
def serve_output(job_id, filename):
    if job_id not in processing_jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404
    # Serve file from job's output directory
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], job_id), filename)

# Function to clean up old jobs
def cleanup_old_jobs():
    now = datetime.now()
    retention = timedelta(hours=app.config['JOB_RETENTION_HOURS'])
    for job_id in list(processing_jobs.keys()):
        created_at = datetime.fromisoformat(processing_jobs[job_id]['created_at'])
        if now - created_at > retention:
            # Delete job output directory
            shutil.rmtree(os.path.join(app.config['OUTPUT_FOLDER'], job_id), ignore_errors=True)
            # Remove job from processing_jobs
            del processing_jobs[job_id]

# Initialize scheduler to run cleanup every hour
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_old_jobs, 'interval', hours=1)
scheduler.start()

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)