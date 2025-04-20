from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for







@app.route('/')
def home():
    return render_template('index.html')  # Render index.html






# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)