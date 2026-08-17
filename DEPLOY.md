# Web demo deploy

The **desktop app** (`python app/main.py`) cannot run on Railway/Render/Vercel. Only the FastAPI web demo is deployed.

## Deploy to Render (recommended)

Railway free 512 MB plus FAST_MODE double-scaling produced wrong pickleball compression. Render uses the same Docker image with pickleball analysis at 960px (long edge) and a single baseline scale.

### 1. Push to GitHub

Repo: `https://github.com/AbdullahYaseen01/Tennis-Table`

### 2. New Web Service

1. Open [dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**
2. Connect GitHub repo **Tennis-Table**
3. Render reads `render.yaml`. Confirm:

| Field | Value |
|---|---|
| **Runtime** | Docker |
| **Branch** | `main` |
| **Instance** | **Starter** ($7/mo). Free 512 MB often OOMs on video |
| **Health check** | `/health` |
| **Do not set** | `VERCEL=1` |

### 3. Deploy

Click **Create Web Service**. First build takes several minutes. When Live:

`https://<your-service>.onrender.com`

- Upload UI: `/`
- Health: `/health` (should include `"platform": "render"`)
- API docs: `/docs`

### Render limits

- Keep videos **under ~20 seconds**, 1080p or smaller, under **40 MB**
- Disk is ephemeral unless you add a Persistent Disk at `/data`
- Select **Pickleball** in the UI and upload the **original** video (not a previously annotated export)

---

## Deploy to Railway

Same Docker image. Auto-deploy from GitHub can stall if **Wait for CI** is on (Vercel/Render GitHub checks). Turn Wait for CI **off**, then **Ctrl+K → Deploy Latest Commit**.

Do **not** set `VERCEL=1`. Free 512 MB: keep videos **10–15 s**, 720p, under **20 MB**.

---

## Deploy to Vercel

Hobby timeout is **10s**, Pro **60s**. Only tiny clips. Full pickleball tests belong on Render.

---

## Local

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```
