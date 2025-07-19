web: uv run gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 main:app
release: uv run python -c "from app import app, db; app.app_context().push(); db.create_all()"