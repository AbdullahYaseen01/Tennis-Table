# Web demo deploy (Vercel)

## Deploy to Vercel

1. Push this repo to GitHub: `https://github.com/AbdullahYaseen01/Tennis-Table`
2. Go to [vercel.com](https://vercel.com) → **Add New Project** → import **Tennis-Table**
3. Framework preset: **Other**
4. Root directory: `.` (repo root)
5. Deploy — Vercel reads `vercel.json` automatically

## What runs on Vercel

- FastAPI web demo at `/` (upload video, live camera UI)
- Endpoints: `/upload`, `/status/{id}`, `/download/{id}`, `/analyze-frame`, `/analyze-baseline`

## Local full app (desktop + all features)

```bash
pip install -r requirements-local.txt
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
python app/main.py   # PySide6 desktop app
```

## Why two requirements files?

- `requirements.txt` — slim deps for Vercel (OpenCV headless + FastAPI only, under 500 MB)
- `requirements-local.txt` — full desktop stack (PySide6, ultralytics, scipy, etc.)

## Vercel limits

- **Function timeout**: 10s (Hobby) / 60s (Pro) — long videos may need Pro or a separate worker
- **No ffmpeg** on Vercel by default — MP4 conversion may fall back to AVI
- **Ephemeral storage** — uploads use `/tmp`; files are not kept after the function ends
- **OpenCV** uses `opencv-python-headless` (see `requirements.txt`)

For production with heavy video processing, consider **Railway**, **Render**, or **Fly.io** with Docker + ffmpeg.
