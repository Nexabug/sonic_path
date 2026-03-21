# Sonic Path

Sonic Path is an assistive navigation prototype that combines live camera input, depth estimation, and directional feedback concepts for obstacle awareness.

## Project Structure

This repo currently contains the Flask prototype:

- `app.py` - Flask server, camera pipeline, MiDaS inference, and API routes
- `templates/index.html` - main frontend used by Flask
- `templates/requirements.txt` - copied dependency reference inside `templates`
- `requirements.txt` - Python dependencies for the app

## What The App Does

- captures live video from a webcam or USB camera
- runs MiDaS depth estimation on frames
- splits the scene into left, center, and right zones
- raises obstacle alerts based on depth intensity
- exposes a live UI with visual guidance, stats, and directional feedback concepts

## Tech Stack

- Python
- Flask
- Flask-CORS
- OpenCV
- PyTorch
- NumPy
- MiDaS

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## API Endpoints

- `/` - main UI
- `/video` - MJPEG video stream
- `/data` - latest alert and zone data
- `/cameras` - list available camera indices
- `/set_camera` - switch the active camera with a `POST` request
- `/health` - health and runtime status

Example request:

```bash
curl -X POST http://localhost:5000/set_camera ^
  -H "Content-Type: application/json" ^
  -d "{\"index\":0}"
```

## Notes

- the default camera index is set in `app.py`
- the app starts background camera and inference threads on startup
- MiDaS model loading may take some time on first run
- CUDA is used automatically when available, otherwise inference runs on CPU

## Hackathon Context

This project is designed as a hackathon build for visually impaired navigation support, using depth-based obstacle awareness along with UI concepts for spatial audio, voice alerts, and haptics.
