ARG BUILD_FROM
FROM $BUILD_FROM

# Install Python dependencies
#RUN apk add --no-cache python3 py3-pip cython3 git curl  \
#  && python3 -m pip install --upgrade pip setuptools wheel
RUN apk update && apk add --no-cache python3 py3-pip cython3 git curl  \

# Copy the add-on code.  Crucially *before* requirements.txt, so Docker layer caching works!
COPY . /app
WORKDIR /app

# Install Python dependencies
# Install sherpa-onnx.  Make sure ARM architecture is handled correctly.
RUN pip3 install --upgrade --no-cache-dir \
    -f https://k2-fsa.github.io/sherpa/onnx/cpu-pip.html \
    sherpa-onnx \
    wyoming

# Copy the files
COPY run.py /app/

# Run the application on port 10400
CMD ["python3", "/app/run.py"]


