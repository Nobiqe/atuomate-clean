# Import necessary libraries for file handling, Flask, threading, scheduling, and utilities
import os
import random
import shutil
import traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from id_card_model import process_id_card
from functools import wraps
import magic  # For validating file MIME types

app = Flask(__name__, static_folder='output_images', static_url_path='/output_images')

app.config['UPLOAD_FOLDER'] = 'uploads'# Directory for uploaded files
app.config['OUTPUT_FOLDER'] = 'output_images'# Directory for processed outputs
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}# Allowed image formats
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'mov', 'webm'}# Allowed video formats
app.config['MAX_IMAGE_SIZE'] = 16 * 1024 * 1024  # 16MB # Max image size: 16MB
app.config['MAX_VIDEO_SIZE'] = 30 * 1024 * 1024  # 30MB # Max video size: 30MB
app.config['JOB_RETENTION_HOURS'] = 24 # Time to retain job data (24 hours)
app.config['API_KEY'] = 'your-secret-api-key' # API key for authentication
MAX_WORKERS = 4 # Max number of concurrent processing threads

@app.route('/')
def home():
    return render_template('index.html')  # Render index.html


@app.route('/api/upload',methods=['POST'])






# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)