FROM python:3.10-slim

# Set up user and environment
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY --chown=user . .

# Expose port
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
