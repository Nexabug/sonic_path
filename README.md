# Sonic Path

Sonic Path now supports two modes:

- GitHub Pages mode: a shareable browser app in [`index.html`](./index.html)
- Local Flask prototype: the original Python + MiDaS experiment in [`app.py`](./app.py)

## Shareable Link

After GitHub Pages deploys, use:

```text
https://nexabug.github.io/sonic_path/
```

This version works in the browser and asks the user for camera permission directly on their own device.

## What Runs on GitHub Pages

The Pages version is fully static and uses:

- TensorFlow.js
- Coco-SSD object detection
- Browser camera access with `getUserMedia`
- Client-side audio beeps and optional voice alerts

Instead of Flask/OpenCV depth inference, the shared web version estimates obstacle risk from:

- object position: left, center, or right
- object size in frame: bigger usually means closer

That makes it suitable for demos and link sharing.

## How To Use The Shared App

1. Open the GitHub Pages link.
2. Tap `Start Camera`.
3. Allow camera permission.
4. Optionally enable audio and voice alerts.
5. Point the camera toward nearby people or objects to trigger warnings.

## Local Flask Prototype

The repo still includes the original local prototype in [`app.py`](./app.py), which uses:

- Flask
- OpenCV
- PyTorch
- MiDaS depth estimation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

## GitHub Pages Deployment

This repo includes:

- [`.github/workflows/deploy-pages.yml`](./.github/workflows/deploy-pages.yml) for Pages deployment
- [`.nojekyll`](./.nojekyll) to avoid Jekyll processing issues

Pushing to `main` triggers a new Pages deployment.

## Important Note

The GitHub Pages version is a browser-first approximation for demo sharing. It does not run the Python MiDaS pipeline in the cloud. Instead, it uses client-side object detection so the shared link works on phones and laptops without a backend.
