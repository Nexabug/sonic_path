# Sonic Path

Sonic Path is a Flask-based computer vision prototype that turns a live camera feed into depth-aware obstacle guidance. It uses MiDaS depth estimation, divides the scene into left, center, and right zones, and exposes real-time data for a frontend that can drive visual, audio, or haptic alerts.

## Features

- Live camera capture with threaded frame reading
- MiDaS depth inference with OpenCV overlay rendering
- Left, center, and right obstacle zone analysis
- MJPEG video stream for the live depth view
- JSON endpoints for status, camera switching, and telemetry
- Frontend-ready alert data for spatial audio or assistive UI

## Tech Stack

- Python
- Flask
- OpenCV
- PyTorch
- MiDaS
- NumPy

## Requirements

- Python 3.10+ recommended
- A connected webcam or USB camera
- Internet access on first run so `torch.hub` can fetch MiDaS assets

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

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
- `/data` - latest depth and alert data as JSON
- `/cameras` - detected camera indices
- `/set_camera` - switch active camera via `POST`
- `/health` - health/status payload

Example camera switch request:

```bash
curl -X POST http://localhost:5000/set_camera ^
  -H "Content-Type: application/json" ^
  -d "{\"index\":0}"
```

## Notes

- The default camera index is currently set in `app.py`.
- The app starts background threads immediately on startup.
- MiDaS model loading can take a little time on the first launch.
- GPU is used automatically when CUDA is available; otherwise it runs on CPU.

## Project Files

- `app.py` - Flask server, camera pipeline, MiDaS inference, and API routes
- `index.html` - frontend UI for the live feed and alerts
- `requirements.txt` - Python dependencies

## Hackathon Context

This project is designed as an assistive navigation prototype for visually impaired users, combining depth perception with directional feedback concepts such as beeps, voice alerts, and haptics.
