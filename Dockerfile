FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user and environment
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/app/cache \
    SENTENCE_TRANSFORMERS_HOME=/home/user/app/cache

WORKDIR $HOME/app

# Pre-create necessary directories with correct permissions
RUN mkdir -p uploads vector_db processed cache && chown -R user:user $HOME/app

USER user

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-download the embedding model to avoid runtime downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy everything
COPY --chown=user . .

# Expose port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
