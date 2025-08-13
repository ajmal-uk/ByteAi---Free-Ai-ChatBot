# YouTube Video Downloader Web App using Flask
# Note: This is a simple example for educational purposes. Downloading videos from YouTube may violate their terms of service.
# Please use responsibly and ensure compliance with applicable laws.
# Requirements: Install Flask and pytube via pip:
# pip install flask pytube

from flask import Flask, request, send_file, render_template_string
from pytube import YouTube
import os

app = Flask(__name__)

# HTML template for the index page
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Video Downloader</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
        form { margin: 20px; }
        input[type="text"] { width: 400px; padding: 10px; }
        input[type="submit"] { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        input[type="submit"]:hover { background: #0056b3; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>YouTube Video Downloader</h1>
    <form method="post">
        <input type="text" name="url" placeholder="Enter YouTube Video URL" required>
        <br><br>
        <input type="submit" value="Download Video">
    </form>
    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    if request.method == 'POST':
        url = request.form.get('url')
        if not url:
            error = "Please enter a valid URL."
            return render_template_string(INDEX_HTML, error=error)
        
        try:
            yt = YouTube(url)
            stream = yt.streams.get_highest_resolution()
            # Download to a temporary file
            filename = stream.download(output_path='downloads', filename=f"{yt.title}.mp4".replace('/', '_').replace('\\', '_'))
            # Send the file for download
            response = send_file(filename, as_attachment=True)
            # Clean up the file after sending
            @response.call_on_close
            def cleanup():
                try:
                    os.remove(filename)
                except Exception:
                    pass
            return response
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template_string(INDEX_HTML, error=error)

if __name__ == '__main__':
    # Create downloads directory if it doesn't exist
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    app.run(debug=True)