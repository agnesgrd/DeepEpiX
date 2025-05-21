<!-- PROJECT LOGO -->
<div align="center">
  <a href="https://github.com/agnesgrd/DeepEpiX">
    <img src="https://github.com/user-attachments/assets/bbe75d9c-204c-4890-8a9b-f8d5131b0032" alt="Logo" width="200" />
  </a>
</div>


<h3 align="center">DeepEpiX</h3>

  <p align="center">
    Software for annotation and automatic spike detection in MEG recordings
    <br />
    <a href="https://github.com/agnes_grd/DeepEpiX"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/agnesgrd/DeepEpiX">View Demo</a>
    &middot;
    <a href="https://github.com/agnesgrd/DeepEpiX/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/agnesgrd/DeepEpiX/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This software is designed for clinicians to annotate raw MEG data and run predictive models for spike detection.  

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python-shield]][Python-url]
* [![Dash][Dash-shield]][Dash-url]
* [![Plotly][Plotly-shield]][Plotly-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- **Data Format**: CTF (dir .ds), 4D Neuroimaging / BTI data (dir), raw FIF (.fif) 
- **Skills**: Basic terminal usage and Python/Docker knowledge

### Fast Installation with Docker

#### 1. Clone the Repository in Your Working Directory  
```bash
git clone https://github.com/agnesgrd/DeepEpiX.git
```

#### 2. Build and run the Docker container with your local data directory:
```bash
cd DeepEpiX
docker build -t deepepix-app .
docker run -p 8050:8050 -v /home/user/DeepEpiX/data:/DeepEpiX/data deepepix-app # Modify this to point your local data path
# Example for Windows
docker run -p 8050:8050 -v //c/Users/pauli/Documents/MEGBase/data/exampleData:/DeepEpiX/data deepepix-app
```

#### 3. Then, open the app in your web browser at:  
[http://localhost:8050/](http://localhost:8050/)  

### Manual Installation for Development Mode

<details>

Follow these steps to install and set up **DeepEpiX**.

#### 1. Clone the Repository in Your Working Directory  
```bash
git clone https://github.com/agnesgrd/DeepEpiX.git
```

#### 2. Set Up the Dash Environment  
Navigate into the `DeepEpiX` directory:  
```bash
cd DeepEpiX
```

and install the virtual environment for running the Dash app:

<details>
  <summary><b>using pip + venv</b></summary>

  ```bash
  python3 -m venv .dashenv
  source .dashenv/bin/activate
  python3 -m pip install -r requirements/requirements-python3.9.txt
  deactivate
  ```
</details>

<details>
  <summary><b>using conda</b></summary>

  ```bash
  conda create --name .dashenv python=3.9
  conda activate .dashenv
  conda install -c conda-forge numpy scipy pandas matplotlib scikit-learn dash plotly Flask-Caching mne dash-bootstrap-components dash-extensions
  conda deactivate
  ```
</details>

<details>
  <summary><b>or manual installation</b></summary>

  Alternatively, you can manually install the core packages listed in:  
  ```
  requirements/requirements-python3.9.in
  ```
</details>

> **Note:** DeepEpiX was developed using **Python 3.9**, so we recommend using this version.  

---

#### 3. Set Up Prediction Model Environments  
If you want to use prediction models, install the required environments:

<details>
  <summary><b>TensorFlow Environment</b></summary>

  **Using pip + venv:**  
  ```bash
  python3 -m venv .tfenv
  source .tfenv/bin/activate
  python3 -m pip install -r requirements/requirements-tfenv.txt
  deactivate
  ```

  **Using conda:**  
  ```bash
  conda create --name .tfenv python=3.9
  conda activate .tfenv
  conda install --file requirements/requirements-tfenv.txt
  conda deactivate
  ```

  Alternatively, install manually from:  
  ```
  requirements/requirements-tfenv.in
  ```
</details>

<details>
  <summary><b>PyTorch Environment</b></summary>

  **Using pip + venv:**  
  ```bash
  python3 -m venv .torchenv
  source .torchenv/bin/activate
  python3 -m pip install -r requirements/requirements-torchenv.txt
  deactivate
  ```

  **Using conda:**  
  ```bash
  conda create --name .torchenv python=3.9
  conda activate .torchenv
  conda install --file requirements/requirements-torchenv.txt
  conda deactivate
  ```

  Alternatively, install manually from:  
  ```
  requirements/requirements-torchenv.in
  ```
</details>

> These environments (`.tfenv` and `.torchenv`) should be in the **DeepEpiX** directory, as they will be referenced when running prediction models.  
> If you choose different environment names, update them in `static/constants.py` under `TENSORFLOW_ENV` and `TORCH_ENV`.

---

#### 4. Run DeepEpiX  
Activate your Dash environment and start DeepEpiX:  
<details>
  <summary><b>using pip + venv</b></summary>

  ```bash
  source .dashenv/bin/activate
  python3 run.py
  ```
</details>

<details>
  <summary><b>using conda</b></summary>

  ```bash
  conda activate .dashenv
  python3 run.py
  ```
</details>

Then, open the app in your web browser at:  
[http://127.0.0.1:8050/](http://127.0.0.1:8050/)  

---

#### 5. You're Ready to Use DeepEpiX! 🎉  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

> For quick access, ensure that your MEG data is placed in the `data` folder within the project directory.

</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[Python-url]: https://www.python.org
[Python-shield]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Dash-url]: https://dash.plotly.com
[Dash-shield]: https://img.shields.io/badge/Dash-red?style=for-the-badge&logo=Dash
[Plotly-url]: https://plotly.com/python
[Plotly-shield]: https://img.shields.io/badge/Plotly-black?style=for-the-badge&logo=Plotly
