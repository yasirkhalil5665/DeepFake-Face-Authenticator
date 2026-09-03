FROM python:3.11-slim

WORKDIR /code

# System deps for Pillow/torchvision image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API needs at runtime — not notebooks/, results/, venv/
COPY app/ ./app/
COPY src/model_builder.py ./src/model_builder.py
COPY models/efficientnet_model.pth ./models/efficientnet_model.pth

ENV FORCE_CPU=1
EXPOSE 8072

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8072"]
