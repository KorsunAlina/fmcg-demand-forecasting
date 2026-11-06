FROM python:3.12-slim

# Системні залежності (libgomp потрібен для багатопоточності LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# Встановлюємо залежності Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код проєкту
COPY src/ /opt/program/src/

# Додаємо робочу директорію до PYTHONPATH, щоб імпорти src.* працювали з будь-якого місця
ENV PYTHONPATH="/opt/program:${PYTHONPATH}"

# Робимо типову точку входу гнучкою
ENTRYPOINT ["python"]