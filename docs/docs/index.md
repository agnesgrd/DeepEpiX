# 🧠 MEG/EEG Signal Preprocessing App

Welcome to the documentation for the Dash-based MEG/EEG preprocessing application.  
This app provides an interactive web interface for loading, preprocessing, annotating and analyzing raw MEG/EEG files, and last but not least: running prediction models.

---

## 🚀 What This App Does

- ✅ Load raw MEG/EEG datasets (`.ds`, `.fif`, or 4D-compatible)
- ✅ Set frequency filtering parameters (resampling, high-pass, low-pass, notch)
- ✅ Detect ECG peaks via channel hinting
- ✅ Drop bad channels
- ✅ Visualize event statistics, power spectral density, topomap, ICA...
- ✅ Display temporal signal with various options
- ✅ Create custom sensors layout (montage)
- ✅ Run prediction models
- ✅ Measure their performances

---

## 🗂 App Structure

The app is structured around **pages**, **layout** and **callbacks**.


## 📖 Docs Navigation

- [📦 Folder Structure](structure.md)
- [🧩 Page: Preprocessing](pages/preprocessing.md)
- [📊 Page: PSD](pages/psd.md)
- [🔄 Callback Glossary](callbacks/index.md)
- [👨‍💻 Developer Setup](dev/setup.md)

---

## 👩‍💻 Who Is This For?

- Developers extending or maintaining the app
- Researchers using the app for MEG/EEG studies
- Contributors improving UI, performance, or adding features

---

## 🛠 Docker Installation (Quick Start)

Clone the Repository in your working directory:
```bash
git clone https://github.com/agnesgrd/DeepEpiX.git
```

Build and run the Docker container with your local data directory:
```bash
cd DeepEpiX
docker build -t deepepix-app .
docker run -p 8050:8050 -v /home/user/DeepEpiX/data:/DeepEpiX/data deepepix-app # Modify this to point your local data path
```

Example of path to point to your data directory for Windows
```bash
docker run -p 8050:8050 -v //c/Users/pauli/Documents/MEGBase/data/exampleData:/DeepEpiX/data deepepix-app
```

Then, open the app in your web browser at:
[http://localhost:8050/](http://localhost:8050/)