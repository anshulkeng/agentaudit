# Alternative to Fly.io: Railway reads this Procfile directly (no Dockerfile
# needed). Railway only runs one process type per service by default, so if
# you want both the API and dashboard on Railway, create two services from
# the same repo and set each one's Start Command to the corresponding line
# below (Railway ignores the process type label in that case).
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
dashboard: streamlit run frontend/app_streamlit.py --server.port=$PORT --server.address=0.0.0.0
