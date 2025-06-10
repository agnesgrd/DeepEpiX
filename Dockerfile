FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /DeepEpiX

COPY requirements/ /DeepEpiX/requirements

RUN apt-get update && apt-get install -y python3-tk

RUN python3 -m venv /.dashenv
RUN python3 -m venv /.tfenv

RUN /.dashenv/bin/pip install --upgrade pip
RUN /.dashenv/bin/pip install -r requirements/requirements-python3.9.txt

RUN OS=$(uname) && ARCH=$(uname -m) && echo "$OS" && echo "$ARCH" && \
    if [ "$OS" = "Darwin" ]; then \
        echo "Detected macOS ($ARCH). Installing Metal-compatible TensorFlow..."; \
        /.tfenv/bin/pip install -r requirements/requirements-tfenv-macos.txt; \
    elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "aarch64" ]; then \
        echo "Detected Linux x86_64. Installing CUDA-compatible TensorFlow..."; \
        /.tfenv/bin/pip install -r requirements/requirements-tfenv-cuda.txt; \
    fi

ENV VIRTUAL_ENV=/.dashenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH=/DeepEpiX/src

COPY ./ /DeepEpiX/

WORKDIR /DeepEpiX/src

EXPOSE 8050

CMD ["/.dashenv/bin/gunicorn", "-w", "25", "-b", "0.0.0.0:8050", "--timeout", "600", "run:server"]