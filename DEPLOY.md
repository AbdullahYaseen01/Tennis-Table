# Web demo deploy

The **desktop app** (`python app/main.py`) cannot run on Railway/Render/Vercel. Only the FastAPI web demo is deployed.

## Deploy to Railway (recommended)

This project is set up for Railway: Docker + ffmpeg + FastAPI. Use this for client video demos.

### 1. Push to GitHub

Repo: `https://github.com/AbdullahYaseen01/Tennis-Table`

### 2. New Railway project

1. Go to [railway.com/new](https://railway.com/new)
2. **Deploy from GitHub repo** → select **Tennis-Table**
3. Railway reads `railway.toml` and builds the `Dockerfile`

### 3. Service settings

| Setting | Value |
|---|---|
| **Plan** | Hobby ($5/mo, includes $5 usage) |
| **Memory** | **1024 MB** minimum (2048 MB if HD videos crash) |
| **Volume** | Add a volume, mount path **`/data`** |

Do **not** set `VERCEL=1`.

Optional env vars:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `/data` | Upload/output/job files |
| `FAST_MODE` | off | Set `1` if the service runs out of RAM |
| `USE_YOLO` | off | Set `1` only if you add ultralytics to `requirements.txt` |

### 4. Open the app

After deploy: `https://<your-service>.up.railway.app`

- Upload UI: `/`
- Health: `/health`
- API docs: `/docs`

Keep test videos short (under ~30 seconds) and under ~50 MB.

---

## Deploy to Render

Render can run long jobs in the background and has `ffmpeg`, so annotated MP4 download works. Vercel times out on long videos.

### 1. Push this repo to GitHub

Your code must be on GitHub (already is if you pushed earlier).

### 2. Create a Web Service

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**
2. Connect the GitHub repo (`Tennis-Table` / `tennis-ball-tester`)
3. Settings:

| Field | Value |
|---|---|
| **Runtime** | Docker (uses the repo `Dockerfile`) |
| **Branch** | `main` |
| **Instance** | **Starter** ($7/mo) — Free (512 MB) often crashes on video |
| **Health check** | `/api` |

If you skip Docker and use native Python instead:

| Field | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build** | `pip install -r requirements.txt` |
| **Start** | `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT` |

Native Python has **no ffmpeg** unless you add an `Aptfile` with `ffmpeg`. Output may stay `.avi`.

### 3. Deploy

Click **Create Web Service**. First build takes a few minutes. When it is Live, open:

`https://<your-service>.onrender.com`

Upload a video at `/`. API docs: `/docs`.

### Render notes

- Do **not** set `VERCEL=1` — that forces slow inline processing.
- Disk is **ephemeral**: uploads/outputs vanish on restart. Add a **Persistent Disk** (mount `/app/data`) if you need files to survive.
- Free tier sleeps after ~15 min idle; first request after sleep is slow.
- Keep videos short (under ~30 s) on Starter RAM.

---

## Deploy to Vercel

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
