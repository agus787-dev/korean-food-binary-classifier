FROM python:3.12-slim

# Hugging Face Spaces runs the container as the user with ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# CPU-only PyTorch: the image is much smaller than the default (CUDA) build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Other dependencies (copied first, so Docker caches this layer when only code changes)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code and trained model. Keep the same folder layout as the project:
# src/ and outputs/models/ (utils.py builds paths relative to the project root)
COPY --chown=user src/ src/
COPY --chown=user app/ app/
COPY --chown=user outputs/models/ outputs/models/

# Hugging Face expects the app on port 7860
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]