# 🐋 Quick Start : Docker Installation (for Production Mode only) 

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

# 🛠 Local Installation (for Development Mode)

[↪️ Go to Developer Guide > Setup & Run](dev/setup.md)