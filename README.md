Automating ID Card Processing with Git: The atuomate-clean Repository
Introduction
The atuomate-clean repository, hosted at github.com/Nobiqe/atuomate-clean, is a Python-based project designed to automate the processing of identification cards using computer vision and optical character recognition (OCR). This article explores the project's structure, its use of Git for version control, and the strategic decision to exclude the large yolov8x.pt model file from the repository, as it is automatically downloaded during setup.
Project Overview
The atuomate-clean project consists of two primary Python scripts: app.py and id_card_model.py. Together, they form a Flask-based web application that processes ID card images and verification videos, extracting relevant information such as names, dates, and serial numbers.
app.py: The Web Application Backend
The app.py script serves as the core of the web application, built using the Flask framework. It handles:

File Uploads: Accepts front and back ID card images and a verification video, enforcing strict validation for file types (PNG, JPG, MP4, etc.) and size limits (16MB for images, 30MB for videos).
Asynchronous Processing: Uses a thread pool (ThreadPoolExecutor) to process uploaded files concurrently, ensuring scalability.
Job Management: Tracks processing jobs with a unique job ID, storing status and results in memory and saving outputs to an output_images directory.
API Security: Implements API key authentication for secure access to upload and status endpoints.
Cleanup: Runs a scheduled task (via APScheduler) to delete old job data after 24 hours, optimizing storage.

The application provides a user-friendly interface with routes for uploading files, viewing results, and serving processed outputs, making it accessible for both manual and programmatic use.
id_card_model.py: ID Card Processing Logic
The id_card_model.py script contains the computer vision and OCR logic for extracting information from ID card images. Key features include:

Object Detection: Uses YOLO models (polov11.pt and yolov8x.pt) to detect ID cards and persons within images.
Text Extraction: Employs EasyOCR for Persian text recognition and the hezar model for enhanced OCR accuracy on printed Persian text.
Text Processing: Normalizes Persian text, converts Persian numbers to English, and formats dates using regular expressions and the hazm library.
Image Preprocessing: Applies advanced image processing techniques (e.g., bilateral filtering, thresholding, and morphological operations) to improve OCR accuracy.
Logging: Saves extracted text to a log file for debugging and validation.

The script is optimized for Persian ID cards, handling challenges like text orientation and varying image quality.
Git Repository Structure
The atuomate-clean repository is structured to be lightweight and modular, with the following key files:

app.py: The Flask application backend.
id_card_model.py: The ID card processing logic.
templates/: HTML templates for the web interface (e.g., index.html, results.html, error.html).
uploads/: Directory for temporarily storing uploaded files.
output_images/: Directory for processed images and videos.
.gitignore: Excludes unnecessary files like myenv/, *.pyc, and large model files.
.gitattributes: Configures Git LFS for tracking large files (if needed).
LICENSE: Specifies the project's licensing terms.
README.md: Provides setup instructions and project overview.

The repository uses Git for version control, enabling efficient collaboration and change tracking.
Managing Large Files: Why yolov8x.pt is Excluded
The yolov8x.pt file, a pre-trained YOLOv8x model weighing approximately 130.55 MB, is a critical component for object detection in id_card_model.py. However, it is intentionally excluded from the Git repository for the following reasons:

Automatic Download: The project is configured to automatically download yolov8x.pt from the Ultralytics YOLO repository or a specified source during setup, reducing the need to store it in the repository.
GitHub File Size Limit: GitHub imposes a 100 MB limit on individual files in standard Git repositories. Including yolov8x.pt would require Git Large File Storage (LFS), which adds complexity and potential costs (GitHub's free LFS limit is 1 GB).
Repository Efficiency: Excluding large files keeps the repository lightweight, making cloning and pulling faster for contributors.
Version Control Best Practices: Large binary files like model weights are better managed outside version control, as they rarely change and can bloat the repository's history.

Instead of including yolov8x.pt, the README.md provides clear instructions for downloading the model during setup, ensuring seamless integration without burdening the repository.
Git Workflow in atuomate-clean
The project follows a standard Git workflow to manage changes:

Initialization: The repository was initialized with git init and connected to the remote at git@github.com:Nobiqe/atuomate-clean.git using SSH.
Committing Changes: Files are staged with git add . and committed with descriptive messages (e.g., git commit -m "Add Flask app and ID card processing").
Pushing to GitHub: Changes are pushed to the main branch using git push -u origin main.
History Management: When issues with large files (like yolov8x.pt) arose, tools like git filter-repo were used to remove them from the commit history, followed by a force push (git push --force).
LFS Configuration: The .gitattributes file is set up to support Git LFS for other large files if needed, but yolov8x.pt is excluded due to its automatic download.

This workflow ensures a clean and maintainable repository, with large files handled externally.
Challenges and Solutions
One challenge encountered was attempting to include yolov8x.pt in early commits, which led to GitHub rejecting pushes due to the 100 MB file size limit. The solution involved:

Using git filter-repo to remove yolov8x.pt from the commit history.
Updating the .gitignore file to prevent accidental inclusion of large model files.
Configuring the project to download yolov8x.pt automatically, as documented in the README.md.

These steps resolved the issue while maintaining repository integrity.
Conclusion
The atuomate-clean repository demonstrates an effective use of Git for managing a computer vision project. By excluding the large yolov8x.pt file and leveraging automatic downloads, the project maintains a lightweight and efficient repository while ensuring all necessary components are accessible. The combination of Flask for web functionality, YOLO for object detection, and OCR for text extraction makes this project a robust solution for automated ID card processing. Developers can clone the repository, follow the setup instructions, and contribute to its development, all while benefiting from a streamlined Git workflow.
For more details, visit github.com/Nobiqe/atuomate-clean.
