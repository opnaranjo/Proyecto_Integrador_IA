# Imagen para desplegar la app en Hugging Face Spaces (Docker) o Render.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8501
WORKDIR /app

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY config.yaml ./
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/raw/ data/raw/
COPY .streamlit/ .streamlit/

# Hugging Face Spaces corre el contenedor con un usuario sin privilegios.
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8501)}/_stcore/health')"
CMD streamlit run app/app.py --server.port=${PORT} --server.address=0.0.0.0
