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
		cython \
		python3-pip \
		python3 \
		&& rm -rf /var/lib/apt/lists/*
#RUN apk update && apk add --no-cache ffmpeg tar curl bzip2 git cython3

# Copy the add-on code.  Crucially *before* requirements.txt, so Docker layer caching works!
COPY . /app
WORKDIR /app

# Install Python dependencies
# Install sherpa-onnx.  Make sure ARM architecture is handled correctly.
#RUN pip3 install --no-cache-dir sherpa-onnx
RUN pip install --break-system-packages --no-cache-dir sherpa-onnx
#-f https://k2-fsa.github.io/sherpa/onnx/cpu.html
#install 
RUN pip install --break-system-packages --no-cache-dir git+https://github.com/rhasspy/wyoming.git

# Copy the files
COPY run.py /app/

# Run the application on port 10400
CMD ["python3", "/app/run.py"]


