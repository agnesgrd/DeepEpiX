FROM python:3.9

# Désactiver les fichiers .pyc et activer le log sans buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définir le répertoire de travail
WORKDIR /DeepEpiX

# Copier uniquement les fichiers nécessaires avant d'installer les dépendances
COPY requirements/ /DeepEpiX/requirements

# Installer les dépendances système et créer un seul environnement virtuel
RUN apt-get update && apt-get install -y python3-tk

# Créer les environnements virtuels
RUN python3 -m venv /.dashenv
# RUN python3 -m venv /.torchenv
RUN python3 -m venv /.tfenv

# Installer les dépendances dans les environnements virtuels respectifs
RUN /.dashenv/bin/pip install -r requirements/requirements-python3.9.txt
# RUN /.torchenv/bin/pip install -r requirements/requirements-torchenv.txt

# Install TensorFlow based on OS and architecture
RUN OS=$(uname) && ARCH=$(uname -m) && echo "$OS" && echo "$ARCH" && \
    if [ "$OS" = "Darwin" ]; then \
        echo "Detected macOS ($ARCH). Installing Metal-compatible TensorFlow..."; \
        /.tfenv/bin/pip install -r requirements/requirements-tfenv-macos.txt; \
    elif [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "aarch64" ]; then \
        echo "Detected Linux x86_64. Installing CUDA-compatible TensorFlow..."; \
        /.tfenv/bin/pip install -r requirements/requirements-tfenv-cuda.txt; \
    fi
    
# Set dashenv as the default environment
ENV VIRTUAL_ENV=/.dashenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copier le reste des fichiers après installation des dépendances
COPY . /DeepEpiX/

# Exposer le port
EXPOSE 8050

CMD ["/.dashenv/bin/gunicorn", "-w", "25", "-b", "0.0.0.0:8050", "--timeout", "600", "run:server"]