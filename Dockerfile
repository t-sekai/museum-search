FROM pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3-venv \
        vim \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8000

CMD ["bash"]
