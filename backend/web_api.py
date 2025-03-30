from flask import Flask, request, jsonify, send_from_directory
import tempfile
import os
import uuid
from footage_analysis import analyze_video

app = Flask(__name__, static_folder="frontend", static_url_path="")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp4")
    video.save(temp_path)

    try:
        results = analyze_video(temp_path)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
