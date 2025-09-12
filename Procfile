web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 60 --workers=1 --threads=1 --max-requests=10 --max-requests-jitter=5 --worker-tmp-dir /dev/shm --preload
