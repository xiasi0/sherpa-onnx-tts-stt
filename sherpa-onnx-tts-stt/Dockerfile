ARG BUILD_FROM "python:3.11.9-slim-bullseye"
FROM $BUILD_FROM

# Install Python dependencies
#RUN apk add --no-cache python3 py3-pip cython3 git curl  \
#  && python3 -m pip install --upgrade pip setuptools wheel
#RUN apk update && apk add --no-cache python3 py3-pip cython3 git curl  \
#RUN apk update && apk add --no-cache git curl py3-pip cython cmake make py3-setuptools py3-wheel g++ gcc  
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        tar \
        curl \
        bzip2 \
		git \
		cython3 \
		python3-pip \
		python3 \
		&& rm -rf /var/lib/apt/lists/*
#RUN apk update && apk add --no-cache ffmpeg tar curl bzip2 git cython3

WORKDIR /
COPY rootfs /

# Create model directories
RUN mkdir -p /stt-models && mkdir -p /tts-models

# Download and extract default STT model (sherpa-onnx-paraformer-zh-2023-03-28)
WORKDIR /stt-models
RUN curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2 -o sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2 && \
    tar -xf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2 && \
    rm sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2 && \
    mv sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx . && \
    mv sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt . && \
    rm -rf sherpa-onnx-paraformer-zh-2023-03-28

# Download and extract default TTS model (matcha-icefall-zh-baker)
WORKDIR /tts-models
RUN curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2 -o matcha-icefall-zh-baker.tar.bz2 && \
    tar -xf matcha-icefall-zh-baker.tar.bz2 && \
    rm matcha-icefall-zh-baker.tar.bz2 && \
    mv matcha-icefall-zh-baker/* . && \
    rm -rf matcha-icefall-zh-baker
RUN curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx -o hifigan_v2.onnx

HEALTHCHECK --start-period=3m --interval=20m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost 10400 \
        | grep -iq "sherpa" \
        || exit 1
# Copy the add-on code.  Crucially *before* requirements.txt, so Docker layer caching works!
COPY . /app
WORKDIR /app

# Install Python dependencies
# Install sherpa-onnx.  Make sure ARM architecture is handled correctly.
#RUN pip3 install --no-cache-dir sherpa-onnx
RUN pip install --break-system-packages --no-cache-dir sherpa-onnx numpy
#-f https://k2-fsa.github.io/sherpa/onnx/cpu.html
#install 
RUN pip install --break-system-packages --no-cache-dir git+https://github.com/rhasspy/wyoming.git

# Copy the files
COPY run.py /app/



