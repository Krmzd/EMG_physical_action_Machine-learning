FROM python:3.11-slim

# OpenMP library that XGBoost needs (the piece missing on your Mac)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# SageMaker starts containers with the command "serve", so create it
RUN printf '#!/bin/sh\nexec uvicorn api:app --host 0.0.0.0 --port 8080\n' > /usr/local/bin/serve \
 && chmod +x /usr/local/bin/serve

EXPOSE 8080
CMD ["serve"]
