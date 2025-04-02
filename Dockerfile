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
RUN python3 -m venv /.torchenv
RUN python3 -m venv /.tfenv

# Installer les dépendances dans les environnements virtuels respectifs
RUN /.dashenv/bin/pip install -r requirements/requirements-python3.9.txt
RUN /.torchenv/bin/pip install -r requirements/requirements-torchenv.txt
RUN /.tfenv/bin/pip install -r requirements/requirements-tfenv.txt

# Set dashenv as the default environment
ENV VIRTUAL_ENV=/.dashenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copier le reste des fichiers après installation des dépendances
COPY . /DeepEpiX/

# Exposer le port
EXPOSE 8080

CMD ["python3", "run.py"]