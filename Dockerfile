FROM continuumio/miniconda3

ENV CONDA_ENV_NAME=minimap-detection

WORKDIR /app/MinimapDetection

COPY . /app/MinimapDetection

# System deps commonly required by OpenCV and image/video tooling
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		libgl1 \
		libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

RUN conda env create -f /app/MinimapDetection/environment.yml

RUN conda run -n ${CONDA_ENV_NAME} --no-capture-output pip install --no-cache-dir \
        torch==2.5.1+cu121 \
        torchvision==0.20.1+cu121 \
        torchaudio==2.5.1+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "minimap-detection", "bash"]
