FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
RUN pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      'torch>=2.7,<3' \
    && pip install '.[postgres,inference]'

COPY alembic ./alembic
COPY docs ./docs
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "mormi_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
