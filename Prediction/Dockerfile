FROM python:3.8-slim-buster
WORKDIR /app
COPY  requirements.txt requirements.txt
COPY  Prediction.py Prediction.py
COPY  run_service.py run_service.py
RUN apt-get update  && apt-get install -y build-essential manpages-dev \   
 && python -m pip install --no-cache-dir --upgrade pip  && python -m pip install --no-cache-dir -r requirements.txt
