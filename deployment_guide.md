# Deployment Guide - Face Recognition Attendance System

This guide will help you deploy your Face Recognition Attendance System to various hosting platforms.

## Platform Deployment Instructions

### 1. Render (Recommended)

**Steps to deploy:**

1. **Create a Render account** at [render.com](https://render.com)

2. **Connect your repository:**
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Choose the repository containing this code

3. **Configure the deployment:**
   - **Build Command:** `pip install uv && uv sync`
   - **Start Command:** `uv run gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 main:app`
   - **Environment:** Python 3.11

4. **Add environment variables:**
   - `DATABASE_URL` - (Render will provide this automatically when you add PostgreSQL)
   - `SESSION_SECRET` - Generate a random secret key

5. **Add PostgreSQL database:**
   - Go to Dashboard → "New" → "PostgreSQL"
   - Name it `face-recognition-db`
   - Connect it to your web service

6. **Deploy:**
   - Click "Create Web Service"
   - Wait for deployment to complete

### 2. Railway

**Steps to deploy:**

1. **Create a Railway account** at [railway.app](https://railway.app)

2. **Deploy from GitHub:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Add PostgreSQL:**
   - Click "New" → "Database" → "Add PostgreSQL"
   - Railway will automatically provide DATABASE_URL

4. **Set environment variables:**
   - Go to your service → "Variables"
   - Add `SESSION_SECRET` with a random secret key
   - Add `PYTHON_VERSION` = `3.11`

5. **Configure deployment:**
   - Railway will automatically use the `railway.json` configuration
   - Build and start commands are pre-configured

### 3. Heroku

**Steps to deploy:**

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create a new Heroku app:**
   ```bash
   heroku create your-app-name
   ```

3. **Add PostgreSQL addon:**
   ```bash
   heroku addons:create heroku-postgresql:essential-0
   ```

4. **Set environment variables:**
   ```bash
   heroku config:set SESSION_SECRET=$(openssl rand -hex 32)
   heroku config:set FLASK_ENV=production
   ```

5. **Deploy:**
   ```bash
   git push heroku main
   ```

### 4. DigitalOcean App Platform

**Steps to deploy:**

1. **Create a DigitalOcean account** at [digitalocean.com](https://digitalocean.com)

2. **Create new app:**
   - Go to App Platform
   - Click "Create App"
   - Connect your GitHub repository

3. **Configure the app:**
   - **Build Command:** `pip install uv && uv sync`
   - **Run Command:** `uv run gunicorn --bind 0.0.0.0:$PORT --workers 2 main:app`

4. **Add database:**
   - Add a PostgreSQL database component
   - DigitalOcean will provide DATABASE_URL automatically

5. **Environment variables:**
   - Add `SESSION_SECRET` with a random value

### 5. Google Cloud Run

**Steps to deploy:**

1. **Enable APIs:**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

2. **Build and deploy:**
   ```bash
   gcloud run deploy face-recognition-attendance \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

3. **Add Cloud SQL PostgreSQL:**
   ```bash
   gcloud sql instances create attendance-db \
     --database-version=POSTGRES_13 \
     --region=us-central1 \
     --tier=db-f1-micro
   ```

## Environment Variables Required

All platforms need these environment variables:

- **DATABASE_URL**: PostgreSQL connection string (usually auto-provided)
- **SESSION_SECRET**: Random secret key for sessions (generate with `openssl rand -hex 32`)

## Important Notes

1. **File Uploads**: Some platforms have ephemeral storage. For persistent file storage, consider:
   - AWS S3 for Heroku/Railway
   - Google Cloud Storage for Google Cloud
   - DigitalOcean Spaces for DigitalOcean

2. **OpenCV Dependencies**: The system automatically installs required system libraries for face recognition.

3. **Memory Requirements**: Face recognition requires at least 512MB RAM. Use appropriate instance sizes.

4. **Database Migrations**: The app automatically creates tables on first run.

## Recommended Platform Comparison

| Platform | Cost | Ease | Performance | PostgreSQL | File Storage |
|----------|------|------|-------------|------------|--------------|
| **Render** | Free tier available | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Free tier | Ephemeral |
| **Railway** | $5/month after trial | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Free tier | Ephemeral |
| **Heroku** | $7/month minimum | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ $9/month | Ephemeral |
| **DigitalOcean** | $12/month minimum | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ $15/month | Persistent |

## Production Optimizations

1. **Enable Redis caching** for better performance
2. **Use CDN** for static assets
3. **Configure SSL/HTTPS** (most platforms auto-enable)
4. **Set up monitoring** and error tracking
5. **Configure backup** for PostgreSQL database

## Testing Deployment

After deployment, test these features:
1. User registration and login
2. Period creation
3. Student enrollment with photo upload
4. Face recognition attendance (both single and multi-face)
5. Batch image processing
6. Report generation and CSV export

## Support

If you encounter issues:
1. Check platform logs for error messages
2. Verify all environment variables are set
3. Ensure PostgreSQL database is connected
4. Check system dependencies for OpenCV/MediaPipe
5. Verify file upload permissions and storage