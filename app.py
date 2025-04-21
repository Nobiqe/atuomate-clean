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
from id_card_processor import process_id_card
from functools import wraps
import magic  # For validating file MIME types

app = Flask(__name__, static_folder='output_images', static_url_path='/output_images')

@app.route('/')
def home():
    return render_template('index.html')  # Render index.html


@app.route('/api/upload',methods=['POST'])






# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)