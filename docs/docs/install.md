## 🐋 Quick Start : Docker Installation (for Production Mode only) 

Clone the Repository in your working directory:
```bash
git clone https://github.com/agnesgrd/DeepEpiX.git
```

Build and Run the Docker Container with your Local Data Directory:
```bash
cd DeepEpiX
docker build -t deepepix-app .
docker run -p 8050:8050 -v /home/user/DeepEpiX/data:/DeepEpiX/data deepepix-app # Modify this to point your local data path
```

Example of Windows Path to Point to your Data Directory:
```bash
docker run -p 8050:8050 -v //c/Users/pauli/Documents/MEGBase/data/exampleData:/DeepEpiX/data deepepix-app
```

Then, open the app in your web browser at:
[http://localhost:8050/](http://localhost:8050/)

You are Ready to Use DeepEpiX ! 🤸‍♂️

---

## 🛠 Local Installation (for Development Mode)

Clone the Repository in your working directory:
```bash
git clone https://github.com/agnesgrd/DeepEpiX.git
```

Set up the Dash Environment:
```bash
cd DeepEpiX
python3 -m venv .dashenv
source .dashenv/bin/activate
python3 -m pip install -r requirements/requirements-python3.9.txt
deactivate
```
🗒️ DeepEpiX was developed using Python 3.9, so we recommend using this version.

Set up Prediction Model Environments: 
```bash
python3 -m venv .tfenv
source .tfenv/bin/activate
python3 -m pip install -r requirements/requirements-tfenv.txt
deactivate
```
```bash
python3 -m venv .torchenv
source .torchenv/bin/activate
python3 -m pip install -r requirements/requirements-torchenv.txt
deactivate
```
Activate your Dash Environment and Start Running the App:
```bash
source .dashenv/bin/activate
python3 src/run.py
```
Then, open the app in your web browser at:
[http://localhost:8050/](http://localhost:8050/)

You can start coding the app while seeing any modifications directly on the dashboard. 🥳

🗒️ For quick access, ensure that your MEG data is placed in the data folder within the project directory.